import random

def agent_random(board, mark):
    return random.choices(range(board.shape[1]))  