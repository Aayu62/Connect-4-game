import numpy as np

#Create an empty board
def create_board(rows,column):
    return np.zeros((rows, column), dtype=int)

#Function to drop pieces
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


#Function to print board
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


def play_game(agent1,agent2,board):
    game_over = False
    turn = 1

    print("GAME STARTS")
    while not game_over:
        print_board(board)
        print("\n"+str(turn))
        if turn == 1:
            col = int(input("Chose your move(1-" + str(board.shape[1])+ "):"))-1
        else:
            col = agent2(board, 2)
        
        if valid_move(board, col):
            drop_pieces(board,col,turn)

            if is_winning(board, turn):
                print_board(board)
                print(f"{"YOU" if turn == 1 else "Agent"} WINS!!")
                break
        else :
            print("Enter a Valid move!!!")
            continue
        turn = turn%2 + 1