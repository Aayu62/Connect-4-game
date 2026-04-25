import random
import math
from .Connect_4 import drop_pieces, valid_move, is_winning


# ──────────────────────────────────────────────
#  Heuristic evaluation (unchanged from before)
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
#  Move ordering: center columns first
#  This is the key enabler for alpha-beta pruning.
#  Better moves explored first → more branches cut.
# ──────────────────────────────────────────────

def get_ordered_moves(board):
    """
    Returns valid columns sorted by distance from center.
    Center-first ordering dramatically improves alpha-beta cutoffs
    because stronger moves are explored first.
    """
    cols = board.shape[1]
    center = cols // 2
    valid_cols = [c for c in range(cols) if valid_move(board, c)]
    # Sort by absolute distance from center (ascending)
    return sorted(valid_cols, key=lambda c: abs(center - c))


# ──────────────────────────────────────────────
#  Minimax with Alpha-Beta Pruning
#
#  Alpha: best score the MAXIMIZER can guarantee so far
#  Beta:  best score the MINIMIZER can guarantee so far
#
#  Pruning rule:
#    - If current node's value >= beta  → maximizer won't pick this path
#      (minimizer above already has something better) → prune (β cut-off)
#    - If current node's value <= alpha → minimizer won't pick this path
#      (maximizer above already has something better) → prune (α cut-off)
#
#  Result: same output as plain minimax, but skips large parts of the tree.
#  At depth 5 with good move ordering this is ~10x faster than plain minimax.
# ──────────────────────────────────────────────

def minimax_ab(board, depth, alpha, beta, maximizing, mark):
    """
    Args:
        board       : current numpy board state
        depth       : remaining search depth
        alpha       : best score maximizer can guarantee (start: -inf)
        beta        : best score minimizer can guarantee (start: +inf)
        maximizing  : True if it's the AI's turn
        mark        : AI's piece value (1 or 2)

    Returns:
        (best_col, best_score)
    """
    opp_mark = 1 if mark == 2 else 2
    valid_cols = get_ordered_moves(board)

    # ── Terminal state checks (ORDER MATTERS) ──
    # Check wins BEFORE depth==0 so we don't miss a winning leaf
    if is_winning(board, mark):
        # Prefer winning sooner → reward higher score at greater depth
        return None, 1_000_000 + depth

    if is_winning(board, opp_mark):
        return None, -(1_000_000 + depth)

    if len(valid_cols) == 0:
        return None, 0  # Draw

    if depth == 0:
        return None, score_position(board, mark)

    # ── Recursive search ──
    best_col = valid_cols[0]  # fallback (always valid due to move ordering)

    if maximizing:
        value = -math.inf

        for col in valid_cols:
            temp_board = board.copy()
            drop_pieces(temp_board, col, mark)
            _, new_score = minimax_ab(temp_board, depth - 1, alpha, beta, False, mark)

            if new_score > value:
                value = new_score
                best_col = col

            alpha = max(alpha, value)

            # Beta cut-off: minimizer above won't allow this path
            if alpha >= beta:
                break

    else:  # minimizing (opponent's turn)
        value = math.inf

        for col in valid_cols:
            temp_board = board.copy()
            drop_pieces(temp_board, col, opp_mark)
            _, new_score = minimax_ab(temp_board, depth - 1, alpha, beta, True, mark)

            if new_score < value:
                value = new_score
                best_col = col

            beta = min(beta, value)

            # Alpha cut-off: maximizer above won't allow this path
            if alpha >= beta:
                break

    return best_col, value

#  Public agent interface (used by views.py)

def minimax_agent(board, mark, depth=5):
    """
    Drop-in replacement for the old minimax_agent.
    Depth 5 with alpha-beta is roughly equivalent in speed
    to depth 4 plain minimax, but plays significantly stronger.
    Raise to depth=6 if you want even stronger play.
    """
    col, _ = minimax_ab(board, depth, -math.inf, math.inf, True, mark)
    return col


def agent_random(board, mark):
    """Kept for data generation and testing purposes."""
    return random.choice(range(board.shape[1]))