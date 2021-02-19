from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.clock import Clock

from Visualize_the_Board.Data_Conversion.chess_coords_to_real_coords import convert_coordinates
from Visualize_the_Board.Data_Conversion.position_of_pieces import position_dic


class BoardScreen(Screen):
    def __init__(self, **kw):
        super(BoardScreen, self).__init__(**kw)

    def display(self, move):
        # Functionality for every move; moving the piece to the correct location and updating the dictionary
        self.ids[position_dic[str(move)[0] + str(move)[1]]].pos = (
            convert_coordinates.to_number()[str(move)[2] + str(move)[3]][0],
            convert_coordinates.to_number()[str(move)[2] + str(move)[3]][1])

        # The ID of the piece that moved
        piece = position_dic[str(move)[0] + str(move)[1]]

        position_dic[str(str(move)[0] + str(move)[1])] = 'None'
        position_dic[str(str(move)[2] + str(move)[3])] = str(piece)

        # Move the Trail
        self.ids["Trail One"].pos = (
            convert_coordinates.to_number()[str(move)[2] + str(move)[3]][0],
            convert_coordinates.to_number()[str(move)[2] + str(move)[3]][1])
        self.ids["Trail Two"].pos = (
            convert_coordinates.to_number()[str(move)[0] + str(move)[1]][0],
            convert_coordinates.to_number()[str(move)[0] + str(move)[1]][1])


# Builds the App
class ChessWindow(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.window = BoardScreen()
        # Clock.schedule_interval(self.window.display('e4'), 1)

    def build(self):
        return self.window
