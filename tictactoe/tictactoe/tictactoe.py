"""
Tic Tac Toe Player
"""

import math
import random
import copy
X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    openspaces = 0
    for i in board:
        for e in i:
            if e == EMPTY:
                openspaces +=1
    if openspaces % 2 == 0:
        return O
    else:
        return X



def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    openspaces = []
    for i in range(len(board)):
        for e in range(len(board[i])):
            if board[i][e] == EMPTY:
                openspaces.append((i,e))
    return openspaces


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    result = copy.deepcopy(board)
    result[action[0]][action[1]] = player(board)
    return result


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    
    for i in range(len(board)):
        dummy = (set(board[i]))
        if len(dummy) == 1 and EMPTY not in dummy:
            if board[i][0]=='X':
                return X
            else:
                return O
    for i in range(len(board)):
        dummy = set([board[0][i],board[1][i],board[2][i]])
        if len(dummy) == 1 and EMPTY not in dummy:
            if board[0][i]=='X':
                return X
            else:
                return O
    dummy = set([board[1][1],board[0][0],board[2][2]])
    dummy1 = set([board[1][1],board[0][2],board[2][0]])
    if (len(dummy) == 1 and EMPTY not in dummy) or (len(dummy1) == 1 and EMPTY not in dummy1):
        if board[1][1]=='X':
            return X
        else:
            return O  
    return None
def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) != None:
        return True
    elif EMPTY in [e for i in board for e in i]:
        return False
    else:
        return True
    


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    vals ={'X':1,'O':-1}
    val = winner(board)
    if val != None:
        return vals[val]
    else:
        return 0

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return (0,utility(board))
    curr = player(board)
    pos_actions = actions(board)
    
    if curr == X:
        best_action = (0,float('-inf'))
        for action in pos_actions:
            opp = minimax(result(board,action))[-1]
            if opp > best_action[-1]:
                best_action = (action, opp)   
                if opp == 1:
                    return best_action 
        return best_action
    else:
        best_action = (0,float('inf'))   
        for action in pos_actions:
            opp = minimax(result(board,action))[-1]
            if opp < best_action[-1]:
                best_action = (action, opp)
                if opp == -1:
                    return best_action
        return best_action

