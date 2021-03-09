from Visualize_the_Board.Board import ChessWindow
from Visualize_the_Board.Board import ChessWindowLinux
from kivy.config import Config
from kivy.core.window import Window


class GUI:
    def __init__(self, is_linux: bool) -> None:
        self.is_linux = is_linux

    def setup(self, player_ai):
        # Set the Height and Width of the Board
        width_of_board = 820
        height_of_board = 800

        # Set the Hight and Width of the App
        Config.set('graphics', 'width', str(width_of_board))
        Config.set('graphics', 'height', str(height_of_board))

        # Make the App non-resizable
        Config.set('graphics', 'resizable', '0')
        Config.write()

        # Make the top Windows bar go away
        Window.borderless = False

        # Disable the Multi-Touch
        Config.set('input', 'mouse', 'mouse,disable_multitouch')

        # Runs the
        if self.is_linux:
            ChessWindowLinux(player_ai).run()
        else:
            ChessWindow(player_ai).run()
