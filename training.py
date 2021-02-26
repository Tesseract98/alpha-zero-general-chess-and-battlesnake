import sys
import threading

from Coach import Coach
from chesspy.ChessGame import ChessGame as Game
# from chesspy.pytorch.NNet import NNetWrapper as nn
from chesspy.tensorflow.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 3,
    'numEps': 3,  # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 5,  #
    'updateThreshold': 0.3, # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 64,  # Number of game examples to train the neural networks.
    'numMCTSSims': 2,  # Number of games moves for MCTS to simulate.
    'arenaCompare': 2,  # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 4,
    'checkpoint': './training/',
    'load_model': False,
    'load_folder_file': ('./training/model/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 2,
})

if __name__ == "__main__":
    sys.setrecursionlimit(int(9e+5))
    threading.stack_size(int(9e+6))

    my_game = Game()
    nnet = nn(my_game)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(my_game, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()

    thread = threading.Thread(target=c.learn())
    thread.start()
    # c.learn()
