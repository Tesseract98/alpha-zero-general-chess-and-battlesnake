import numpy as np

from MCTS import MCTS
from Visualize_the_Board.visualize_main import GUI
from chesspy.ChessGame import ChessGame as Game
from chesspy.ChessPlayers import RandomPlayer, StaticChessPlayer
from chesspy.pytorch.NNet import NNetWrapper as NNet
from utils import *


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
        n1.load_checkpoint('./training/', 'temp.pth.tar')
        args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
        mcts1 = MCTS(g, n1, args1)
        ap = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
    except ValueError:
        print('warning: no AI found')
        ap = None

    return ap
    # return rp


if __name__ == "__main__":
    visualize = GUI()
    visualize.setup(init_player())
