import numpy as np

def create_board(rows, column):
    return np.zeros((rows, column), dtype=int)

def drop_pieces(board, column, mark):
    for row in range(board.shape[0]-1, -1, -1):
        if board[row][column] == 0:
            board[row][column] = mark
            return True
    return False

def valid_move(board, col):
    for row in range(board.shape[0]-1, -1, -1):
        if board[row][col] == 0:
            return True
    return False

def is_draw(board):
    return all(board[0][c] != 0 for c in range(board.shape[1]))

def print_board(board):
    print(board)
    print("\n")

def is_winning(board, mark):
    #Horizontal
    for row in range(board.shape[0]):
        for col in range(board.shape[1]-3):
            window = [board[row][col+i] for i in range(4)]
            if window.count(mark) == 4:
                return True
    
    #vertical
    for row in range(board.shape[0]-3):
        for col in range(board.shape[1]):
            window = [board[row+i][col] for i in range(4)]
            if window.count(mark) == 4:
                return True
    
    #\
    for row in range(board.shape[0]-3):
        for col in range(board.shape[1]-3):
            window = [board[row+i][col+i] for i in range(4)]
            if window.count(mark) == 4:
                return True
    
    #/
    for row in range(board.shape[0]-3):
        for col in range(3, board.shape[1]):
            window = [board[row+i][col-i] for i in range(4)]
            if window.count(mark) == 4:
                return True
    return False