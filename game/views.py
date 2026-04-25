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
    return JsonResponse({"message": "Game started", "board": board_list})


def restart_game(request):
    board = create_board(ROWS, COLUMNS)
    request.session['board'] = board.tolist()
    return JsonResponse({"message": "Game restarted", "board": board.tolist()})


def player_move(request):
    data = json.loads(request.body)
    col = data.get("col")
    board = np.array(request.session.get('board'))

    if not valid_move(board, col):
        return JsonResponse({"error": "Invalid move"})

    drop_pieces(board, col, 1)

    if is_winning(board, 1):
        request.session['board'] = board.tolist()
        return JsonResponse({"board": board.tolist(), "winner": "You"})

    if is_draw(board):
        request.session['board'] = board.tolist()
        return JsonResponse({"board": board.tolist(), "winner": "Draw"})

    agent_col = minimax_agent(board, 2)
    drop_pieces(board, agent_col, 2)

    if is_winning(board, 2):
        request.session['board'] = board.tolist()
        return JsonResponse({"board": board.tolist(), "winner": "AI"})

    if is_draw(board):
        request.session['board'] = board.tolist()
        return JsonResponse({"board": board.tolist(), "winner": "Draw"})

    request.session['board'] = board.tolist()
    return JsonResponse({"board": board.tolist(), "agent_move": agent_col})