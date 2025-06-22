import tkinter as tk
from agent import *
from Connect_4 import *
import numpy as np

ROWS = 6
COLUMNS = 7
board = create_board(ROWS, COLUMNS)

root = tk.Tk()
root.title("Connect Four")

PLAYER_COLORS = {1: "red", 2: "yellow"}
EMPTY_COLOR = "lightblue"

frame = tk.Frame(root)
frame.pack()

labels = [[None for _ in range(COLUMNS)] for _ in range(ROWS)]

def draw_board():
    for r in range(ROWS):
        for c in range(COLUMNS):
            cell = board[r][c]
            color = PLAYER_COLORS.get(cell, EMPTY_COLOR)
            labels[r][c].config(bg=color)

def make_move(col):
    global current_player, board

    if valid_move(board, col):
        drop_pieces(board, col, current_player)
        draw_board()
        
        if is_winning(board, current_player):
            status_label.config(text=f"Player {current_player} wins!")
            disable_all_buttons()
            return
        
        current_player = 2
        root.after(500, agent_move)

def agent_move():
    global current_player, board
    col = agent_minimax(board, current_player)
    drop_pieces(board, col, current_player)
    draw_board()

    if is_winning(board, current_player):
        status_label.config(text=f"Agent (Player {current_player}) wins!")
        disable_all_buttons()
    else:
        current_player = 1

def restart_game():
    global board, current_player
    board = create_board(ROWS, COLUMNS)
    current_player = 1
    draw_board()
    status_label.config(text="Your Turn (Player 1)")
    for btn in column_buttons:
        btn.config(state="normal")

def disable_all_buttons():
    for btn in column_buttons:
        btn.config(state="disabled")

column_buttons = []
for c in range(COLUMNS):
    btn = tk.Button(root, text=f"Drop\n{c+1}", command=lambda col=c: make_move(col))
    btn.pack(side=tk.LEFT, padx=2, pady=2)
    column_buttons.append(btn)

for r in range(ROWS):
    for c in range(COLUMNS):
        label = tk.Label(frame, text=" ", width=4, height=2, relief="ridge", bg=EMPTY_COLOR)
        label.grid(row=r, column=c, padx=1, pady=1)
        labels[r][c] = label

status_label = tk.Label(root, text="Your Turn (Player 1)", font=("Helvetica", 14))
status_label.pack(pady=10)

restart_btn = tk.Button(root, text="Restart Game", command=restart_game, bg="lightgreen", font=("Helvetica", 12))
restart_btn.pack(pady=5)

current_player = 1
draw_board()

root.mainloop()