"""
generate_data.py  —  Stage 2 (v2): Training Data Generation
=============================================================
Key fix over v1: canonical board encoding.

Every board state is now encoded from the CURRENT PLAYER's perspective:
  Current player's piece → +1
  Opponent's piece       → -1
  Empty                  →  0

And every label is +1 if the current player eventually won, -1 if they lost.

This means the model learns "is MY position good?" rather than
"is Player 1's position good?" — making it player-agnostic.
"""

import numpy as np
import math
import random
import os
import time

#Board logic
def create_board():
    return np.zeros((6, 7), dtype=int)

def drop_pieces(board, col, mark):
    for row in range(5, -1, -1):
        if board[row][col] == 0:
            board[row][col] = mark
            return True
    return False

def valid_move(board, col):
    return board[0][col] == 0

def get_valid_cols(board):
    return [c for c in range(7) if valid_move(board, c)]

def is_winning(board, mark):
    for r in range(6):
        for c in range(4):
            if all(board[r][c+i] == mark for i in range(4)): return True
    for r in range(3):
        for c in range(7):
            if all(board[r+i][c] == mark for i in range(4)): return True
    for r in range(3):
        for c in range(4):
            if all(board[r+i][c+i] == mark for i in range(4)): return True
    for r in range(3):
        for c in range(3, 7):
            if all(board[r+i][c-i] == mark for i in range(4)): return True
    return False

def score_position(board, mark):
    score = 0
    score += list(board[:, 3]).count(mark) * 3
    opp = 1 if mark == 2 else 2
    for r in range(6):
        row = list(board[r, :])
        for c in range(4):
            w = row[c:c+4]
            if w.count(mark) == 4:   score += 100
            elif w.count(mark) == 3 and w.count(0) == 1: score += 5
            elif w.count(mark) == 2 and w.count(0) == 2: score += 2
            if w.count(opp) == 3 and w.count(0) == 1:   score -= 4
    return score

#Minimax with alpha-beta
def minimax_ab(board, depth, alpha, beta, maximizing, mark):
    opp = 1 if mark == 2 else 2
    valid_cols = sorted(get_valid_cols(board), key=lambda c: abs(3 - c))

    if is_winning(board, mark):   return None, 1_000_000 + depth
    if is_winning(board, opp):    return None, -(1_000_000 + depth)
    if not valid_cols:            return None, 0
    if depth == 0:                return None, score_position(board, mark)

    best_col = valid_cols[0]
    if maximizing:
        value = -math.inf
        for col in valid_cols:
            tmp = board.copy(); drop_pieces(tmp, col, mark)
            _, s = minimax_ab(tmp, depth-1, alpha, beta, False, mark)
            if s > value: value, best_col = s, col
            alpha = max(alpha, value)
            if alpha >= beta: break
    else:
        value = math.inf
        for col in valid_cols:
            tmp = board.copy(); drop_pieces(tmp, col, opp)
            _, s = minimax_ab(tmp, depth-1, alpha, beta, True, mark)
            if s < value: value, best_col = s, col
            beta = min(beta, value)
            if alpha >= beta: break

    return best_col, value

def agent_move(board, mark, depth=3):
    col, _ = minimax_ab(board, depth, -math.inf, math.inf, True, mark)
    return col

#Canonical board encoding
def encode_board(board, current_player):
    """
    Encode board from the CURRENT PLAYER's perspective.
      Current player's piece  -> +1.0
      Opponent's piece        -> -1.0
      Empty                   ->  0.0

    This is canonical encoding — the model always sees itself as +1.
    A P2 board is flipped so P2's pieces appear as +1.
    The model learns "is MY position good?" regardless of which player it is.
    """
    opp = 2 if current_player == 1 else 1
    encoded = np.zeros(42, dtype=np.float32)
    for r in range(6):
        for c in range(7):
            idx = r * 7 + c
            if board[r][c] == current_player:
                encoded[idx] =  1.0
            elif board[r][c] == opp:
                encoded[idx] = -1.0
    return encoded

#Single game simulation
def play_one_game(depth=3, noise_prob=0.15):
    """
    Play one full game of minimax vs minimax with noise for variety.

    Returns list of (encoded_state, label) where:
      encoded_state : 42-float canonical encoding (from mover's perspective)
      label         : +1 if current player eventually won, -1 if lost, 0 draw
    """
    board  = create_board()
    turn   = 1
    record = []   # list of (encoded_board, which_player_moved)

    while True:
        valid_cols = get_valid_cols(board)
        if not valid_cols:
            return [(state, 0.0) for state, _ in record]

        # Encode from current player's perspective BEFORE the move
        record.append((encode_board(board, current_player=turn), turn))

        # Move: minimax or random noise
        col = random.choice(valid_cols) if random.random() < noise_prob \
              else agent_move(board, turn, depth)

        drop_pieces(board, col, turn)

        if is_winning(board, turn):
            # Winner's states get +1, loser's states get -1
            labeled = []
            for state, mover in record:
                if mover == turn:
                    labeled.append((state,  1.0))
                else:
                    labeled.append((state, -1.0))
            return labeled

        turn = 2 if turn == 1 else 1

#Main generation loop
def generate_dataset(n_games=3000, depth=3, save_path="data/connect4_dataset.npz"):
    os.makedirs("data", exist_ok=True)

    all_states = []
    all_labels = []
    wins = losses = draws = 0

    print(f"\n{'='*55}")
    print(f"  Generating {n_games} self-play games (depth={depth})")
    print(f"  Canonical encoding: each player sees itself as +1")
    print(f"{'='*55}\n")

    t_start = time.time()

    for game_idx in range(n_games):
        samples = play_one_game(depth=depth)

        for state, label in samples:
            all_states.append(state)
            all_labels.append(label)

        last_label = samples[-1][1]
        if   last_label ==  1.0: wins   += 1
        elif last_label == -1.0: losses += 1
        else:                    draws  += 1

        if (game_idx + 1) % 200 == 0:
            elapsed   = time.time() - t_start
            rate      = (game_idx + 1) / elapsed
            remaining = (n_games - game_idx - 1) / rate
            print(f"  [{game_idx+1:>4}/{n_games}]  "
                  f"states: {len(all_states):>6}  |  "
                  f"W/L/D: {wins}/{losses}/{draws}  |  "
                  f"ETA: {remaining:.0f}s")

    X = np.array(all_states, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)

    np.savez_compressed(save_path, X=X, y=y)

    elapsed = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Total positions : {len(X):,}")
    print(f"  +1 (won)  : {int((y== 1).sum()):,}")
    print(f"  -1 (lost) : {int((y==-1).sum()):,}")
    print(f"   0 (draw) : {int((y== 0).sum()):,}")
    print(f"  Saved to  : {save_path}")
    print(f"{'='*55}\n")
    return X, y

def inspect_dataset(path="data/connect4_dataset.npz"):
    data = np.load(path)
    X, y = data['X'], data['y']
    print(f"\nDataset: {path}")
    print(f"  X shape : {X.shape}")
    print(f"  Labels  :  +1={int((y==1).sum()):,}  -1={int((y==-1).sum()):,}  0={int((y==0).sum()):,}")
    print(f"  X range : [{X.min():.1f}, {X.max():.1f}]\n")

if __name__ == "__main__":
    X, y = generate_dataset(n_games=3000)
    inspect_dataset()