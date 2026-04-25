"""
benchmark.py  —  Run from your project root:
    python benchmark.py

Compares old plain minimax vs new alpha-beta at the same depth
so you can see the concrete speedup Stage 1 gives you.
"""
import time
import math
import random
import numpy as np

# ── Paste-in of old minimax (no changes) ──────────────────────────────────────

def create_board(rows=6, cols=7):
    return np.zeros((rows, cols), dtype=int)

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

def is_winning(board, mark):
    for row in range(board.shape[0]):
        for col in range(board.shape[1]-3):
            if all(board[row][col+i] == mark for i in range(4)):
                return True
    for row in range(board.shape[0]-3):
        for col in range(board.shape[1]):
            if all(board[row+i][col] == mark for i in range(4)):
                return True
    for row in range(board.shape[0]-3):
        for col in range(board.shape[1]-3):
            if all(board[row+i][col+i] == mark for i in range(4)):
                return True
    for row in range(board.shape[0]-3):
        for col in range(3, board.shape[1]):
            if all(board[row+i][col-i] == mark for i in range(4)):
                return True
    return False

def score_position(board, mark):
    score = 0
    rows, cols = board.shape
    center = cols // 2
    score += list(board[:, center]).count(mark) * 3
    return score  # simplified for speed comparison

def minimax_old(board, depth, maximizing, mark):
    valid_cols = [c for c in range(board.shape[1]) if valid_move(board, c)]
    opp_mark = 1 if mark == 2 else 2
    if depth == 0 or len(valid_cols) == 0:
        return None, score_position(board, mark)
    if is_winning(board, mark):
        return None, 1_000_000
    if is_winning(board, opp_mark):
        return None, -1_000_000
    best_col = random.choice(valid_cols)
    if maximizing:
        value = -math.inf
        for col in valid_cols:
            temp = board.copy()
            drop_pieces(temp, col, mark)
            _, s = minimax_old(temp, depth-1, False, mark)
            if s > value:
                value = s
                best_col = col
        return best_col, value
    else:
        value = math.inf
        for col in valid_cols:
            temp = board.copy()
            drop_pieces(temp, col, opp_mark)
            _, s = minimax_old(temp, depth-1, True, mark)
            if s < value:
                value = s
                best_col = col
        return best_col, value

# ── New alpha-beta ────────────────────────────────────────────────────────────

def get_ordered_moves(board):
    cols = board.shape[1]
    center = cols // 2
    valid_cols = [c for c in range(cols) if valid_move(board, c)]
    return sorted(valid_cols, key=lambda c: abs(center - c))

def minimax_ab(board, depth, alpha, beta, maximizing, mark):
    opp_mark = 1 if mark == 2 else 2
    valid_cols = get_ordered_moves(board)
    if is_winning(board, mark):
        return None, 1_000_000 + depth
    if is_winning(board, opp_mark):
        return None, -(1_000_000 + depth)
    if len(valid_cols) == 0:
        return None, 0
    if depth == 0:
        return None, score_position(board, mark)
    best_col = valid_cols[0]
    if maximizing:
        value = -math.inf
        for col in valid_cols:
            temp = board.copy()
            drop_pieces(temp, col, mark)
            _, s = minimax_ab(temp, depth-1, alpha, beta, False, mark)
            if s > value:
                value = s
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = math.inf
        for col in valid_cols:
            temp = board.copy()
            drop_pieces(temp, col, opp_mark)
            _, s = minimax_ab(temp, depth-1, alpha, beta, True, mark)
            if s < value:
                value = s
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
    return best_col, value

# ── Benchmark ────────────────────────────────────────────────────────────────

def run_benchmark(depth=5, n_positions=3):
    print(f"\n{'='*55}")
    print(f"  Connect 4 — Minimax Benchmark  (depth={depth})")
    print(f"{'='*55}\n")

    # Use a mid-game board so the tree isn't trivially small
    board = create_board()
    moves = [3, 3, 2, 4, 3, 3, 2]  # a few opening moves
    for i, col in enumerate(moves):
        drop_pieces(board, col, (i % 2) + 1)

    print("Board state used for benchmark:")
    print(board, "\n")

    times_old = []
    times_new = []

    for i in range(n_positions):
        # Old minimax
        t0 = time.perf_counter()
        col_old, _ = minimax_old(board, depth, True, 2)
        t1 = time.perf_counter()
        times_old.append(t1 - t0)

        # New alpha-beta
        t0 = time.perf_counter()
        col_new, _ = minimax_ab(board, depth, -math.inf, math.inf, True, 2)
        t1 = time.perf_counter()
        times_new.append(t1 - t0)

    avg_old = sum(times_old) / n_positions
    avg_new = sum(times_new) / n_positions
    speedup = avg_old / avg_new if avg_new > 0 else float('inf')

    print(f"  Old minimax (depth {depth}):   {avg_old:.3f}s  → col {col_old}")
    print(f"  Alpha-beta  (depth {depth}):   {avg_new:.3f}s  → col {col_new}")
    print(f"\n  🚀 Speedup: {speedup:.1f}x faster\n")
    print(f"  At depth {depth+1} old would take ~{avg_old*7:.1f}s")
    print(f"  At depth {depth+1} alpha-beta takes ~{avg_new*3:.1f}s (estimated)")
    print(f"\n{'='*55}\n")

if __name__ == "__main__":
    run_benchmark(depth=5)