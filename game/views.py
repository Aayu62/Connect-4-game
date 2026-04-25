import json
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from .Connect_4 import create_board, drop_pieces, valid_move, is_winning, is_draw
from .agent import minimax_agent


ROWS = 6
COLUMNS = 7


def home(request):
    return render(request, "game/index.html")


def start_game(request):
    board = create_board(ROWS, COLUMNS)
    board_list = board.tolist()
    request.session['board'] = board_list
    request.session['ai_first'] = False  # clear ai_first flag
    return JsonResponse({"message": "Game started", "board": board_list})


def restart_game(request):
    board = create_board(ROWS, COLUMNS)
    request.session['board'] = board.tolist()
    request.session['ai_first'] = False  # clear ai_first flag
    return JsonResponse({"message": "Game restarted", "board": board.tolist()})


def ai_first_move(request):
    """Start a new game and immediately let the AI (mark=1) play first."""
    board = create_board(ROWS, COLUMNS)
    agent_col = minimax_agent(board, 1)
    drop_pieces(board, agent_col, 1)
    request.session['board'] = board.tolist()
    request.session['ai_first'] = True
    return JsonResponse({"board": board.tolist(), "agent_move": agent_col})


def player_move(request):
    data = json.loads(request.body)
    col = data.get("col")
    board = np.array(request.session.get('board'))
    ai_first = request.session.get('ai_first', False)

    # When AI went first: player=2, ai=1
    player_mark = 2 if ai_first else 1
    ai_mark     = 1 if ai_first else 2

    if not valid_move(board, col):
        return JsonResponse({"error": "Invalid move"})

    drop_pieces(board, col, player_mark)

    if is_winning(board, player_mark):
        request.session['board'] = board.tolist()
        return JsonResponse({"board": board.tolist(), "winner": "You"})

    if is_draw(board):
        request.session['board'] = board.tolist()
        return JsonResponse({"board": board.tolist(), "winner": "Draw"})

    agent_col = minimax_agent(board, ai_mark)
    drop_pieces(board, agent_col, ai_mark)

    if is_winning(board, ai_mark):
        request.session['board'] = board.tolist()
        return JsonResponse({"board": board.tolist(), "winner": "AI"})

    if is_draw(board):
        request.session['board'] = board.tolist()
        return JsonResponse({"board": board.tolist(), "winner": "Draw"})

    request.session['board'] = board.tolist()
    return JsonResponse({"board": board.tolist(), "agent_move": agent_col})