import math
import numpy as np

np.random.seed(11)
EPS = 1e-8


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

        self.moves = []

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        s = self.game.stringRepresentation(canonicalBoard)

        if self.game.getGameEnded(canonicalBoard, 1) != 0 or s in self.Es and self.Es[s] != 0:
            valids = self.game.getValidMoves(canonicalBoard, 1)
            # print("arena push not ended game")
            if np.sum(valids) > 0:
                # print("random move")
                bestA = np.random.choice(np.where(valids)[0])
                probs = [0] * self.game.getActionSize()
                probs[bestA] = 1
                return probs

        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard, 0)

        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestA = int(np.argmax(counts))
            probs = [0] * len(counts)
            probs[bestA] = 1

            valids = self.game.getValidMoves(self.game.getCanonicalForm(canonicalBoard, 1), 1)
            if valids[np.argmax(probs)] == 0:
                print(bestA, np.argmax(probs))
                print(s)
                print(canonicalBoard.fen())
                print(canonicalBoard)
                # print(self.Qsa)
                print(self.Nsa)
                print(self.Ns)
                # print(self.Ps)
                print(self.Es)
                print(self.Vs)
                raise RuntimeError("something wrong")

            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard, n_iter, rec_limit: int = 1000):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(self.game.toArray(canonicalBoard))
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get
                # overfitting or something else. If you have got dozens or hundreds of these messages you should pay
                # attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        best_act_u_dict = dict()
        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                best_act_u_dict[a] = u

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act

        # this helps to avoid the recursion of a step forward and backward
        # TODO: develop a more productive way to avoid stack overflow
        best_act_u_dict.pop(a)
        for act in self.moves:
            best_act_u_dict_len = len(best_act_u_dict.keys())
            if self.moves.count(act) > 10 and best_act_u_dict_len > 0:
                # self.Ps[s] = self.Ps[s] + valids
                # self.Ps[s] /= np.sum(self.Ps[s])
                possible_moves = list(dict(sorted(best_act_u_dict.items(), key=lambda item: item[1])).keys())
                possible_moves = possible_moves[best_act_u_dict_len // 2:]
                # a = np.random.choice(np.where(valids)[0])
                if len(possible_moves) != 0:
                    # a = np.random.choice(possible_moves)
                    a = possible_moves[-1]

                self.clean_up_moves()
                break

        self.moves.append(a)

        next_s, next_player, move = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        if n_iter > rec_limit:
            # print("!!!EXCEED RECURSION!!!")
            self.Ps[s], v = self.nnet.predict(self.game.toArray(canonicalBoard))
            return v

        v = self.search(next_s, n_iter + 1)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        self.clean_up_moves()
        return -v

    def clean_up_moves(self):
        self.moves = []

    # def clean_all_fields(self):
    #     self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
    #     self.Nsa = {}  # stores #times edge s,a was visited
    #     self.Ns = {}  # stores #times board s was visited
    #     self.Ps = {}  # stores initial policy (returned by neural net)
    #
    #     self.Es = {}  # stores game.getGameEnded ended for board s
    #     self.Vs = {}  # stores game.getValidMoves for board s
    #
    #     self.moves = []
