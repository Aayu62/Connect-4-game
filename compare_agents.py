"""
compare_agents.py  —  Heuristic vs ML Agent head-to-head
=========================================================
Run from your project root:
    python compare_agents.py
"""

import numpy as np
import math
import random
import torch
import torch.nn as nn
import os

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

#ValueNet + canonical encoding
class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(42, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),    nn.Tanh()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def load_valuenet(path="data/value_net.pt"):
    net  = ValueNet()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    return net

def encode_board(board, current_player):
    """Canonical: current player always sees itself as +1."""
    opp = 2 if current_player == 1 else 1
    enc = np.zeros(42, dtype=np.float32)
    for r in range(6):
        for c in range(7):
            v = board[r][c]
            enc[r*7+c] = 1.0 if v == current_player else (-1.0 if v == opp else 0.0)
    return enc

def ml_eval(board, mark, model):
    """Score from current player's perspective — no sign flip needed."""
    t = torch.tensor(encode_board(board, mark), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return model(t).item() * 100.0

#Minimax 
def get_ordered_moves(board):
    return sorted(get_valid_cols(board), key=lambda c: abs(3 - c))

def minimax(board, depth, alpha, beta, maximizing, mark, eval_fn):
    opp = 1 if mark == 2 else 2
    valid_cols = get_ordered_moves(board)

    if is_winning(board, mark):   return None,  1_000_000 + depth
    if is_winning(board, opp):    return None, -(1_000_000 + depth)
    if not valid_cols:            return None, 0
    if depth == 0:                return None, eval_fn(board, mark)

    best_col = valid_cols[0]
    if maximizing:
        value = -math.inf
        for col in valid_cols:
            tmp = board.copy(); drop_pieces(tmp, col, mark)
            _, s = minimax(tmp, depth-1, alpha, beta, False, mark, eval_fn)
            if s > value: value, best_col = s, col
            alpha = max(alpha, value)
            if alpha >= beta: break
    else:
        value = math.inf
        for col in valid_cols:
            tmp = board.copy(); drop_pieces(tmp, col, opp)
            _, s = minimax(tmp, depth-1, alpha, beta, True, mark, eval_fn)
            if s < value: value, best_col = s, col
            beta = min(beta, value)
            if alpha >= beta: break

    return best_col, value


def play_match(eval_p1, eval_p2, depth=4, n_games=10):
    p1_wins = p2_wins = draws = 0
    for game in range(n_games):
        board = create_board()
        turn  = 1
        result = "Draw"
        while True:
            if not get_valid_cols(board):
                draws += 1; break
            eval_fn = eval_p1 if turn == 1 else eval_p2
            col, _  = minimax(board, depth, -math.inf, math.inf, True, turn, eval_fn)
            drop_pieces(board, col, turn)
            if is_winning(board, turn):
                if turn == 1: p1_wins += 1; result = "P1 wins"
                else:         p2_wins += 1; result = "P2 wins"
                break
            turn = 2 if turn == 1 else 1
        print(f"    Game {game+1:>2}: {result}")
    return p1_wins, p2_wins, draws

if __name__ == "__main__":
    if not os.path.exists("data/value_net.pt"):
        print("ERROR: data/value_net.pt not found. Run train_model.py first.")
        exit(1)

    model  = load_valuenet()
    DEPTH  = 4
    GAMES  = 20

    heuristic = lambda board, mark: score_position(board, mark)
    ml        = lambda board, mark: ml_eval(board, mark, model)

    print(f"\n{'='*55}")
    print(f"  Head-to-Head: Heuristic vs ML Agent")
    print(f"  Depth={DEPTH}, {GAMES} games total")
    print(f"{'='*55}")

    print(f"\n  Round 1: ML (P1) vs Heuristic (P2)")
    print(f"  {'─'*35}")
    w1, l1, d1 = play_match(ml, heuristic, DEPTH, GAMES // 2)

    print(f"\n  Round 2: Heuristic (P1) vs ML (P2)")
    print(f"  {'─'*35}")
    w2, l2, d2 = play_match(heuristic, ml, DEPTH, GAMES // 2)

    # ML perspective
    ml_wins   = w1 + l2
    ml_losses = l1 + w2
    ml_draws  = d1 + d2

    print(f"\n{'='*55}")
    print(f"  Results — ML perspective ({GAMES} games)")
    print(f"  Wins  : {ml_wins}")
    print(f"  Losses: {ml_losses}")
    print(f"  Draws : {ml_draws}")
    print(f"  Win Rate: {100*ml_wins/GAMES:.1f}%")
    print(f"{'='*55}\n")