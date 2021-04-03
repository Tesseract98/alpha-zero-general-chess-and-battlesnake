from tqdm import tqdm
import numpy as np
import copy


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=print):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

        # self.unique_player1 = copy.copy(player1)
        # self.unique_player2 = copy.copy(player2)

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        result = 0
        with tqdm(total=2500, position=0, leave=True, desc="episode of a game") as pbar:
            while result == 0:
                it += 1
                if verbose:
                    assert self.display
                    print("Turn ", str(it), "Player ", str(curPlayer))
                    self.display(board)
                action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

                valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

                if valids[action] == 0:
                    print(action)
                    assert valids[action] > 0
                board, curPlayer, _ = self.game.getNextState(board, curPlayer, action)
                result = self.game.getGameEnded(board, curPlayer)
                pbar.update(1)
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(result))
            self.display(board)
        return result

    def playGames(self, num, verbose=False, training=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        print('Arena.playGames', num)

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), position=0, leave=True, desc="Arena.playGames (1)"):
            # if training:
            #     self.init_mcts()
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), position=0, leave=True, desc="Arena.playGames (2)"):
            # if training:
            #     self.init_mcts(is_reversed=True)
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws

    # def init_mcts(self, is_reversed: bool = False):
    #     # print("reset")
    #     self.unique_player1.clean_all_fields()
    #     self.unique_player2.clean_all_fields()
    #     if not is_reversed:
    #         self.player1 = lambda x: np.argmax(self.unique_player1.getActionProb(x, temp=0))
    #         self.player2 = lambda x: np.argmax(self.unique_player2.getActionProb(x, temp=0))
    #     else:
    #         self.player1 = lambda x: np.argmax(self.unique_player2.getActionProb(x, temp=0))
    #         self.player2 = lambda x: np.argmax(self.unique_player1.getActionProb(x, temp=0))
