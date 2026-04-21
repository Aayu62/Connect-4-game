import random
from .Connect_4 import *
import math

def agent_random(board, mark):
    return random.choices(range(board.shape[1]))  

def evaluate_window(window, mark):
    score = 0
    opp_mark = 1 if mark == 2 else 2

    if window.count(mark) == 4:
        score += 100
    elif window.count(mark) == 3 and window.count(0) == 1:
        score += 5
    elif window.count(mark) == 2 and window.count(0) == 2:
        score += 2

    if window.count(opp_mark) == 3 and window.count(0) == 1:
        score -= 4

    return score

def score_position(board, mark):
    score = 0
    rows, cols = board.shape

    # Center column preference
    center = cols // 2
    center_array = list(board[:, center])
    score += center_array.count(mark) * 3

    # Horizontal
    for r in range(rows):
        row_array = list(board[r, :])
        for c in range(cols - 3):
            score += evaluate_window(row_array[c:c+4], mark)

    # Vertical
    for c in range(cols):
        col_array = list(board[:, c])
        for r in range(rows - 3):
            score += evaluate_window(col_array[r:r+4], mark)

    # Diagonal \
    for r in range(rows - 3):
        for c in range(cols - 3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, mark)

    # Diagonal /
    for r in range(rows - 3):
        for c in range(3, cols):
            window = [board[r+i][c-i] for i in range(4)]
            score += evaluate_window(window, mark)

    return score


def minimax(board, depth, maximizing, mark):
    valid_cols = [c for c in range(board.shape[1]) if valid_move(board, c)]
    opp_mark = 1 if mark == 2 else 2

    if depth == 0 or len(valid_cols) == 0:
        return None, score_position(board, mark)

    if is_winning(board, mark):
        return None, 1000000
    if is_winning(board, opp_mark):
        return None, -1000000

    if maximizing:
        value = -math.inf
        best_col = random.choice(valid_cols)

        for col in valid_cols:
            temp_board = board.copy()
            drop_pieces(temp_board, col, mark)
            new_score = minimax(temp_board, depth-1, False, mark)[1]

            if new_score > value:
                value = new_score
                best_col = col

        return best_col, value

    else:
        value = math.inf
        best_col = random.choice(valid_cols)

        for col in valid_cols:
            temp_board = board.copy()
            drop_pieces(temp_board, col, opp_mark)
            new_score = minimax(temp_board, depth-1, True, mark)[1]

            if new_score < value:
                value = new_score
                best_col = col

        return best_col, value
    
def minimax_agent(board, mark):
    col, _ = minimax(board, depth=4, maximizing=True, mark=mark)
    return col



def smart_agent(obs, config):
    valid_moves = valid_moves(obs, config)
    # 1. Try to win
    for col in valid_moves:
        temp_board = drop_pieces(obs.board, col, obs.mark, config)
        if is_winning_move(temp_board, obs.mark, config):
            return col
    # 2. Try to block opponent
    opp_mark = 1 if obs.mark == 2 else 2
    for col in valid_moves:
        temp_board = drop_pieces(obs.board, col, opp_mark, config)
        if is_winning_move(temp_board, opp_mark, config):
            return col
    # 3. Pick center if available
    if config.columns // 2 in valid_moves:
        return config.columns // 2
    # 4. Otherwise, pick random
    return random.choice(valid_moves)

def is_winning_move(board, mark, config):
    for r in range(config.rows):
        for c in range(config.columns):
            idx = r * config.columns + c
            # Horizontal
            if c + config.inarow <= config.columns:
                if all(board[idx + i] == mark for i in range(config.inarow)):
                    return True
            # Vertical
            if r + config.inarow <= config.rows:
                if all(board[idx + i * config.columns] == mark for i in range(config.inarow)):
                    return True
            # Diagonal /
            if c + config.inarow <= config.columns and r - config.inarow >= -1:
                if all(board[idx + i * (config.columns - 1)] == mark for i in range(config.inarow)):
                    return True
            # Diagonal \
            if c + config.inarow <= config.columns and r + config.inarow <= config.rows:
                if all(board[idx + i * (config.columns + 1)] == mark for i in range(config.inarow)):
                    return True
    return False
