# Alpha Zero General (any game, any framework!)

A simplified, highly flexible, commented and (hopefully) easy to understand implementation of self-play based reinforcement learning based on the AlphaGo Zero paper (Silver et al). It is designed to be easy to adopt for any two-player turn-based adversarial game and any deep learning framework of your choice. A sample implementation has been provided for the game of Othello in PyTorch, Keras, TensorFlow and Chainer. An accompanying tutorial can be found [here](http://web.stanford.edu/~surag/posts/alphazero.html). We also have implementations for GoBang and TicTacToe.

To use a game of your choice, subclass the classes in ```Game.py``` and ```NeuralNet.py``` and implement their functions. Example implementations for Othello can be found in ```othello/OthelloGame.py``` and ```othello/{pytorch,keras,tensorflow,chainer}/NNet.py```. 

```Coach.py``` contains the core training loop and ```MCTS.py``` performs the Monte Carlo Tree Search. The parameters for the self-play can be specified in ```main.py```. Additional neural network parameters are in ```othello/{pytorch,keras,tensorflow,chainer}/NNet.py``` (cuda flag, batch size, epochs, learning rate etc.). 

To start training a model for Othello:
```bash
python train.py
```
Choose your framework and game in ```train.py```.

### Docker Installation
For easy environment setup, we can use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Once you have nvidia-docker set up, we can then simply run:
```
./setup_env.sh
```
to set up a (default: pyTorch) Jupyter docker container. We can now open a new terminal and enter:
```
docker exec -ti pytorch_notebook python main.py
```

### Experiments
We trained a PyTorch model for 6x6 Othello (~80 iterations, 100 episodes per iteration and 25 MCTS simulations per turn). This took about 3 days on an NVIDIA Tesla K80. The pretrained model (PyTorch) can be found in ```pretrained_models/othello/pytorch/```. You can play a game against it using ```pit.py```. Below is the performance of the model against a random and a greedy baseline with the number of iterations.
![alt tag](https://github.com/suragnair/alpha-zero-general/raw/master/pretrained_models/6x6.png)

A concise description of our algorithm can be found [here](https://github.com/suragnair/alpha-zero-general/raw/master/pretrained_models/writeup.pdf).

### Contributing
While the current code is fairly functional, we could benefit from the following contributions:
* Game logic files for more games that follow the specifications in ```Game.py```, along with their neural networks
* Neural networks in other frameworks
* Pre-trained models for different game configurations
* An asynchronous version of the code- parallel processes for self-play, neural net training and model comparison. 
* Asynchronous MCTS as described in the paper

### Contributors and Credits
* [Shantanu Thakoor](https://github.com/ShantanuThakoor) and [Megha Jhunjhunwala](https://github.com/jjw-megha) helped with core design and implementation.
* [Shantanu Kumar](https://github.com/SourKream) contributed TensorFlow and Keras models for Othello.
* [Evgeny Tyurin](https://github.com/evg-tyurin) contributed rules and a trained model for TicTacToe.
* [MBoss](https://github.com/1424667164) contributed rules and a model for GoBang.
* [Jernej Habjan](https://github.com/JernejHabjan) contributed RTS game.

Thanks to [pytorch-classification](https://github.com/bearpaw/pytorch-classification) and [progress](https://github.com/verigak/progress).

### Additions of chess game

```
pip install -r requirements.txt
```

Added chess game (```chesspy```) with pytorch / tensorflow AI.

To start training a model for Chess, need to run ```train.py```.

To play with AI need to run ```Play_Agent.py```.

If you want to play with AI, you'll need to extract files from ```best.pth.tar.rar``` archive in training\model\.

The best model were trained (https://colab.research.google.com/notebooks/intro.ipynb) with following parameters:
* 'numIters': 11,
* 'numEps': 15,
* 'tempThreshold': 25,
* 'updateThreshold': 0.5,
* 'maxlenOfQueue': 4096
* 'numMCTSSims': 15,
* 'arenaCompare': 4,
* 'cpuct': 1,
* 'numItersForTrainExamplesHistory': 5.

Also changed MCTS.getActionProb and MCTS.search algorithms, 
they were crashed the program cause of stack overflow (algorithm made repeatable steps and didn't exit out of recursion).

Also added playing visualization in ```Play_Agent.py``` 
(To play with AI, you need to extract the best model from the rar archive in /training/model/).

![alt text](https://github.com/Tesseract98/alpha-zero-general-chess-and-battlesnake/raw/master/data/chess_board.png)
