import numpy as np
import sys

from MCTS import MCTS
from Visualize_the_Board.visualize_main import GUI
from chesspy.ChessGame import ChessGame as Game
from chesspy.ChessPlayers import RandomPlayer, StaticChessPlayer
from chesspy.tensorflow.NNet import NNetWrapper as NNet
# from chesspy.pytorch.NNet import NNetWrapper as NNet
from utils import dotdict


def init_player():
    """
    use this script to play manually with the best temp agent.
    """
    g = Game()

    rp = RandomPlayer(g).play
    sp = StaticChessPlayer(g).play

    try:
        # nnet players
        n1 = NNet(g)
        n1.load_checkpoint('./training/model/', 'best.pth.tar')
        args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
        mcts1 = MCTS(g, n1, args1)
        ap = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
    except ValueError:
        print('warning: no AI found')
        ap = None

    return ap
    # return rp


if __name__ == "__main__":
    sys.setrecursionlimit(5000)
    visualize = GUI(is_linux=True)
    visualize.setup(init_player())
