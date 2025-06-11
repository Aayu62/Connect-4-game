from Connect_4 import *
from agent import *

print("Enter the dimension of connect-Four Game Board")
Rows = int(input("Rows:"))
Column = int(input("Column:"))

board = create_board(Rows,Column)

play_game(None,agent_random,board)