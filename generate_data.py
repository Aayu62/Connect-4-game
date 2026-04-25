"""
Run from your project root:
    python generate_data.py

What it does:
  - Plays minimax (depth 3) vs minimax (depth 3) for N games
  - Records every board state seen during each game
  - Labels each state with the final outcome:
      +1  → Player 1 won
      -1  → Player 1 lost (Player 2 won)
       0  → Draw
  - Saves dataset to data/connect4_dataset.npz

Why depth 3 for generation (not 5)?
  - We're generating thousands of games, speed matters
  - Depth 3 still produces non-random, strategically meaningful positions
  - The NN learns board structure, not just random noise

Board encoding used for training:
  - Player 1 pieces  →  +1
  - Player 2 pieces  →  -1
  - Empty cells      →   0
  - Flattened to a 42-element vector
"""

import numpy as np
import math
import random
import os
import time

#Board logic (standalone, no Django imports needed)
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
    # Horizontal
    for r in range(6):
        for c in range(4):
            if all(board[r][c+i] == mark for i in range(4)):
                return True
    # Vertical
    for r in range(3):
        for c in range(7):
            if all(board[r+i][c] == mark for i in range(4)):
                return True
    # Diagonal \
    for r in range(3):
        for c in range(4):
            if all(board[r+i][c+i] == mark for i in range(4)):
                return True
    # Diagonal /
    for r in range(3):
        for c in range(3, 7):
            if all(board[r+i][c-i] == mark for i in range(4)):
                return True
    return False

def score_position(board, mark):
    """Lightweight heuristic for data generation agent."""
    score = 0
    center_array = list(board[:, 3])
    score += center_array.count(mark) * 3

    for r in range(6):
        row = list(board[r, :])
        for c in range(4):
            w = row[c:c+4]
            opp = 1 if mark == 2 else 2
            if w.count(mark) == 4:   score += 100
            elif w.count(mark) == 3 and w.count(0) == 1: score += 5
            elif w.count(mark) == 2 and w.count(0) == 2: score += 2
            if w.count(opp) == 3 and w.count(0) == 1:    score -= 4
    return score

#Minimax with alpha-beta (same logic as agent.py)
def minimax_ab(board, depth, alpha, beta, maximizing, mark):
    opp = 1 if mark == 2 else 2
    valid_cols = sorted(get_valid_cols(board), key=lambda c: abs(3 - c))

    if is_winning(board, mark):
        return None, 1_000_000 + depth
    if is_winning(board, opp):
        return None, -(1_000_000 + depth)
    if not valid_cols:
        return None, 0
    if depth == 0:
        return None, score_position(board, mark)

    best_col = valid_cols[0]

    if maximizing:
        value = -math.inf
        for col in valid_cols:
            tmp = board.copy()
            drop_pieces(tmp, col, mark)
            _, s = minimax_ab(tmp, depth-1, alpha, beta, False, mark)
            if s > value:
                value, best_col = s, col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = math.inf
        for col in valid_cols:
            tmp = board.copy()
            drop_pieces(tmp, col, opp)
            _, s = minimax_ab(tmp, depth-1, alpha, beta, True, mark)
            if s < value:
                value, best_col = s, col
            beta = min(beta, value)
            if alpha >= beta:
                break

    return best_col, value

def agent_move(board, mark, depth=3):
    col, _ = minimax_ab(board, depth, -math.inf, math.inf, True, mark)
    return col

#Board encoding
def encode_board(board):
    """
    Encode board as a 42-float vector from Player 1's perspective.
      Player 1 piece  → +1.0
      Player 2 piece  → -1.0
      Empty           →  0.0
    """
    encoded = np.zeros(42, dtype=np.float32)
    for r in range(6):
        for c in range(7):
            idx = r * 7 + c
            if board[r][c] == 1:
                encoded[idx] = 1.0
            elif board[r][c] == 2:
                encoded[idx] = -1.0
    return encoded

#Single game simulation

def play_one_game(depth=3, add_noise_prob=0.15):
    """
    Play one full game of minimax vs minimax.

    add_noise_prob: probability of making a random move instead of minimax.
    This adds variety to the dataset so we don't just get the same game
    over and over (minimax is deterministic without noise).

    Returns:
        states  : list of encoded board vectors (one per move)
        outcome : +1 (P1 wins), -1 (P2 wins), 0 (draw)
    """
    board = create_board()
    states = []
    turn = 1  #Player 1 goes first

    while True:
        valid_cols = get_valid_cols(board)

        if not valid_cols:
            outcome = 0  # Draw
            break

        # Record state BEFORE the move
        states.append(encode_board(board))

        # Choose move: minimax or random noise
        if random.random() < add_noise_prob:
            col = random.choice(valid_cols)
        else:
            col = agent_move(board, turn, depth)

        drop_pieces(board, col, turn)

        if is_winning(board, turn):
            outcome = 1 if turn == 1 else -1
            break

        turn = 2 if turn == 1 else 1

    return states, outcome

#Main generation loop

def generate_dataset(n_games=3000, depth=3, save_path="data/connect4_dataset.npz"):
    os.makedirs("data", exist_ok=True)

    all_states  = []
    all_labels  = []

    wins   = 0
    losses = 0
    draws  = 0

    print(f"\n{'='*55}")
    print(f"  Generating {n_games} self-play games (depth={depth})")
    print(f"  This will take a few minutes...")
    print(f"{'='*55}\n")

    t_start = time.time()

    for game_idx in range(n_games):
        states, outcome = play_one_game(depth=depth)

        # Label every state in this game with the final outcome
        for state in states:
            all_states.append(state)
            all_labels.append(outcome)

        # Track outcomes
        if outcome == 1:   wins   += 1
        elif outcome == -1: losses += 1
        else:               draws  += 1

        # Progress update every 200 games
        if (game_idx + 1) % 200 == 0:
            elapsed = time.time() - t_start
            rate = (game_idx + 1) / elapsed
            remaining = (n_games - game_idx - 1) / rate
            print(f"  [{game_idx+1:>4}/{n_games}]  "
                  f"states: {len(all_states):>6}  |  "
                  f"W/L/D: {wins}/{losses}/{draws}  |  "
                  f"ETA: {remaining:.0f}s")

    # Convert to numpy arrays
    X = np.array(all_states,  dtype=np.float32)  # shape: (N, 42)
    y = np.array(all_labels,  dtype=np.float32)  # shape: (N,)

    # Save
    np.savez_compressed(save_path, X=X, y=y)

    elapsed = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Total positions : {len(X):,}")
    print(f"  P1 wins  (+1)   : {wins}  games")
    print(f"  P2 wins  (-1)   : {losses} games")
    print(f"  Draws    ( 0)   : {draws}  games")
    print(f"  Saved to        : {save_path}")
    print(f"{'='*55}\n")

    return X, y

#Quick sanity check
def inspect_dataset(path="data/connect4_dataset.npz"):
    """Load and print basic stats about the saved dataset."""
    data = np.load(path)
    X, y = data['X'], data['y']

    print(f"\nDataset: {path}")
    print(f"  X shape : {X.shape}  (positions × 42 features)")
    print(f"  y shape : {y.shape}")
    print(f"  Labels  :  +1={int((y==1).sum())}  -1={int((y==-1).sum())}  0={int((y==0).sum())}")
    print(f"  X range : [{X.min():.1f}, {X.max():.1f}]\n")


if __name__ == "__main__":
    X, y = generate_dataset(n_games=3000)
    inspect_dataset()