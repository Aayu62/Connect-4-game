import json
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from .Connect_4 import *
from .agent import minimax_agent
from django.views.decorators.csrf import csrf_exempt


ROWS = 6
COLUMNS = 7


def home(request):
    return render(request, "game/index.html")


def start_game(request):
    board = create_board(ROWS, COLUMNS)

    # convert to list (important)
    board_list = board.tolist()

    request.session['board'] = board_list

    return JsonResponse({
        "message": "Game started",
        "board": board_list
    })

@csrf_exempt
def restart_game(request):
    board = create_board(ROWS, COLUMNS)
    request.session['board'] = board.tolist()

    return JsonResponse({
        "message": "Game restarted",
        "board": board.tolist()
    })

@csrf_exempt
def player_move(request):
    import json
    data = json.loads(request.body)
    col = data.get("col")

    import numpy as np
    board = np.array(request.session.get('board'))

    # Player move
    if not valid_move(board, col):
        return JsonResponse({"error": "Invalid move"})

    drop_pieces(board, col, 1)

    if is_winning(board, 1):
        request.session['board'] = board.tolist()
        return JsonResponse({"board": board.tolist(), "winner": "You"})

    # AI move
    agent_col = minimax_agent(board, 2)
    drop_pieces(board, agent_col, 2)

    if is_winning(board, 2):
        request.session['board'] = board.tolist()
        return JsonResponse({"board": board.tolist(), "winner": "AI"})

    request.session['board'] = board.tolist()

    return JsonResponse({
        "board": board.tolist(),
        "agent_move": agent_col
    })