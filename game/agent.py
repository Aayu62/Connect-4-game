"""
game/agent.py  —  Stage 4 (v2): ML-powered Minimax with canonical encoding
===========================================================================
Fix over v1: encode_board now always encodes from the CURRENT player's
perspective, matching how the model was trained.
"""

import random
import math
import os
import numpy as np
import torch
import torch.nn as nn
from .Connect_4 import drop_pieces, valid_move, is_winning

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


#Singleton model loader
_model = None

def load_model(path="data/value_net.pt"):
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(path):
        print(f"[agent] WARNING: {path} not found. Using heuristic fallback.")
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    net = ValueNet()
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    _model = net
    print(f"[agent] ValueNet loaded from {path}")
    return _model


#Canonical board encoding
def encode_board(board, current_player):
    """
    Encode from the CURRENT PLAYER's perspective — always sees itself as +1.
    Must match generate_data.py encoding exactly.
      Current player's piece  -> +1.0
      Opponent's piece        -> -1.0
      Empty                   ->  0.0
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


#Evaluation
def evaluate_board(board, mark, model):
    """
    Score the board for the player whose turn it is (mark).
    Model output is already from current player's perspective (+1 = good for me).
    No sign flip needed — canonical encoding handles perspective automatically.
    """
    if model is not None:
        enc    = encode_board(board, current_player=mark)
        tensor = torch.tensor(enc, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            score = model(tensor).item()
        return score * 100.0   # scale to match win/loss magnitude
    else:
        return score_position(board, mark)


#Heuristic fallback
def evaluate_window(window, mark):
    score = 0
    opp = 1 if mark == 2 else 2
    if window.count(mark) == 4:       score += 100
    elif window.count(mark) == 3 and window.count(0) == 1: score += 5
    elif window.count(mark) == 2 and window.count(0) == 2: score += 2
    if window.count(opp) == 3 and window.count(0) == 1:   score -= 4
    return score

def score_position(board, mark):
    score = 0
    rows, cols = board.shape
    score += list(board[:, cols//2]).count(mark) * 3
    for r in range(rows):
        row = list(board[r, :])
        for c in range(cols - 3):
            score += evaluate_window(row[c:c+4], mark)
    for c in range(cols):
        col = list(board[:, c])
        for r in range(rows - 3):
            score += evaluate_window(col[r:r+4], mark)
    for r in range(rows - 3):
        for c in range(cols - 3):
            score += evaluate_window([board[r+i][c+i] for i in range(4)], mark)
    for r in range(rows - 3):
        for c in range(3, cols):
            score += evaluate_window([board[r+i][c-i] for i in range(4)], mark)
    return score


#Move ordering
def get_ordered_moves(board, mark=None, model=None):
    cols = board.shape[1]
    valid_cols = [c for c in range(cols) if valid_move(board, c)]

    if model is not None and mark is not None and len(valid_cols) > 1:
        scores = []
        for col in valid_cols:
            tmp = board.copy()
            drop_pieces(tmp, col, mark)
            enc = encode_board(tmp, current_player=mark)
            t   = torch.tensor(enc, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                s = model(t).item()
            scores.append((col, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scores]
    else:
        center = cols // 2
        return sorted(valid_cols, key=lambda c: abs(center - c))


#Minimax with alpha-beta
def minimax_ab(board, depth, alpha, beta, maximizing, mark, model):
    opp = 1 if mark == 2 else 2
    valid_cols = get_ordered_moves(board, mark=mark, model=model)

    if is_winning(board, mark):   return None,  1_000_000 + depth
    if is_winning(board, opp):    return None, -(1_000_000 + depth)
    if not valid_cols:            return None, 0
    if depth == 0:                return None, evaluate_board(board, mark, model)

    best_col = valid_cols[0]
    if maximizing:
        value = -math.inf
        for col in valid_cols:
            tmp = board.copy(); drop_pieces(tmp, col, mark)
            _, s = minimax_ab(tmp, depth-1, alpha, beta, False, mark, model)
            if s > value: value, best_col = s, col
            alpha = max(alpha, value)
            if alpha >= beta: break
    else:
        value = math.inf
        for col in valid_cols:
            tmp = board.copy(); drop_pieces(tmp, col, opp)
            _, s = minimax_ab(tmp, depth-1, alpha, beta, True, mark, model)
            if s < value: value, best_col = s, col
            beta = min(beta, value)
            if alpha >= beta: break

    return best_col, value


#Public interface
def minimax_agent(board, mark, depth=5):
    model = load_model()
    col, _ = minimax_ab(board, depth, -math.inf, math.inf, True, mark, model)
    return col

def agent_random(board, mark):
    return random.choice(range(board.shape[1]))