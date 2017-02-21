"""
Reference:
http://dhconnelly.com/paip-python/docs/paip/othello.html
"""

import random

### Board representation
EMPTY, BLACK, WHITE, OUTER = '.', 'X', 'o', '?'
PIECES = (EMPTY, BLACK, WHITE, OUTER)
PLAYERS = {BLACK: 'Black', WHITE: 'White'}

UP, DOWN, LEFT, RIGHT = -10, 10, -1, 1
UP_RIGHT, DOWN_RIGHT, DOWN_LEFT, UP_LEFT = -9, 11, 9, -11
DIRECTIONS = (UP, UP_RIGHT, RIGHT, DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT, UP_LEFT)

def squares():
    """List all the valid squares on the board."""
    return [i for i in xrange(11, 89) if 1 <= (i % 10) <= 8]

def initial_board():
    """Create a new board with the initial black and white positions filled."""
    board = [OUTER] * 100
    for i in squares():
        board[i] = EMPTY
    # The middle four squares should hold the initial piece positions.
    board[44], board[45] = WHITE, BLACK
    board[54], board[55] = BLACK, WHITE
    global count_move
    count_move = 0
    return board

def print_board(board):
    """Get a string representation of the board."""
    rep = ''
    rep += '  %s\n' % ' '.join(map(str, range(1, 9)))
    for row in xrange(1, 9):
        begin, end = 10*row + 1, 10*row + 9
        rep += '%d %s\n' % (row, ' '.join(board[begin:end]))
    return rep

### Checking moves 

def is_valid(move):
    """Is move a square on the board?"""
    return isinstance(move, int) and move in squares()

def opponent(player):
    """Get player's opponent piece."""
    return BLACK if player is WHITE else WHITE

def find_bracket(square, player, board, direction):
    """
    Find a square that forms a bracket with `square` for `player` in the given
    `direction`.  Returns None if no such square exists.
    """
    bracket = square + direction
    if board[bracket] == player:
        return None
    opp = opponent(player)
    while board[bracket] == opp:
        bracket += direction
    return None if board[bracket] in (OUTER, EMPTY) else bracket

def is_legal(move, player, board):
    """Is this a legal move for the player?"""
    hasbracket = lambda direction: find_bracket(move, player, board, direction)
    return board[move] == EMPTY and any(map(hasbracket, DIRECTIONS))

### Making moves

def make_move(move, player, board):
    """Update the board to reflect the move by the specified player."""
    board[move] = player
    for d in DIRECTIONS:
        make_flips(move, player, board, d)
    return board

def make_flips(move, player, board, direction):
    """Flip pieces in the given direction as a result of the move by player."""
    bracket = find_bracket(move, player, board, direction)
    if not bracket:
        return
    square = move + direction
    while square != bracket:
        board[square] = player
        square += direction

### Monitoring players

class IllegalMoveError(Exception):
    def __init__(self, player, move, board):
        self.player = player
        self.move = move
        self.board = board
    
    def __str__(self):
        return '%s cannot move to square %d' % (PLAYERS[self.player], self.move)

def legal_moves(player, board):
    """Get a list of all legal moves for player."""
    return [sq for sq in squares() if is_legal(sq, player, board)]

def any_legal_move(player, board):
    """Can player make any moves?"""
    return any(is_legal(sq, player, board) for sq in squares())

### Putting it all together

# Each round consists of:
#
# - Get a move from the current player.
# - Apply it to the board.
# - Switch players.  If the game is over, get the final score.

def play(black_strategy, white_strategy):
    """Play a game of Othello and return the final board and score."""
    board = initial_board()
    player = BLACK
    strategy = lambda who: black_strategy if who == BLACK else white_strategy
    while player is not None:
        move = get_move(strategy(player), player, board)
        make_move(move, player, board)
        player = next_player(board, player)
    return board, score(BLACK, board)

def next_player(board, prev_player):
    """Which player should move next?  Returns None if no legal moves exist."""
    opp = opponent(prev_player)
    if any_legal_move(opp, board):
        return opp
    elif any_legal_move(prev_player, board):
        return prev_player
    return None

def get_move(strategy, player, board):
    """Call strategy(player, board) to get a move."""
    copy = list(board) # copy the board to prevent cheating
    move = strategy(player, copy)
    if not is_valid(move) or not is_legal(move, player, board):
        raise IllegalMoveError(player, move, copy)
    global count_move
    count_move += 1
    print 'Number of moves: ', count_move    
    return move

def score(player, board):
    """Compute player's score (number of player's pieces minus opponent's)."""
    mine, theirs = 0, 0
    opp = opponent(player)
    for sq in squares():
        piece = board[sq]
        if piece == player: mine += 1
        elif piece == opp: theirs += 1
    return mine - theirs

## Play strategies
### Random
def random_strategy(player, board):
    """A strategy that always chooses a random legal move."""
    return random.choice(legal_moves(player, board))


### Local maximization
def maximizer(evaluate):
    """
    Construct a strategy that chooses the best move by maximizing
    evaluate(player, board) over all boards resulting from legal moves.
    """
    def strategy(player, board):
        def score_move(move):
            return evaluate(player, make_move(move, player, list(board)))
        return max(legal_moves(player, board), key=score_move)
    return strategy

SQUARE_WEIGHTS = [
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0, 120, -20,  20,   5,   5,  20, -20, 120,   0,
    0, -20, -40,  -5,  -5,  -5,  -5, -40, -20,   0,
    0,  20,  -5,  15,   3,   3,  15,  -5,  20,   0,
    0,   5,  -5,   3,   3,   3,   3,  -5,   5,   0,
    0,   5,  -5,   3,   3,   3,   3,  -5,   5,   0,
    0,  20,  -5,  15,   3,   3,  15,  -5,  20,   0,
    0, -20, -40,  -5,  -5,  -5,  -5, -40, -20,   0,
    0, 120, -20,  20,   5,   5,  20, -20, 120,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
]

# A strategy constructed as `maximizer(weighted_score)`, then, will always
# return the move that results in the largest immediate *weighted* gain in
# pieces.

def weighted_score(player, board):
    """
    Compute the difference between the sum of the weights of player's
    squares and the sum of the weights of opponent's squares.
    """
    opp = opponent(player)
    total = 0
    for sq in squares():
        if board[sq] == player:
            total += SQUARE_WEIGHTS[sq]
        elif board[sq] == opp:
            total -= SQUARE_WEIGHTS[sq]
    return total

### Minimax search
""" http://en.wikipedia.org/wiki/Minimax """

def minimax(player, board, depth, evaluate):
    """
    Find the best legal move for player, searching to the specified depth.
    Returns a tuple (move, min_score), where min_score is the guaranteed minimum
    score achievable for player if the move is made.
    """

    # We define the value of a board to be the opposite of its value to our
    # opponent, computed by recursively applying `minimax` for our opponent.
    def value(board):
        return -minimax(opponent(player), board, depth-1, evaluate)[0]
    
    # When depth is zero, don't examine possible moves--just determine the value
    # of this board to the player.
    if depth == 0:
        return evaluate(player, board), None
    
    # We want to evaluate all the legal moves by considering their implications
    # `depth` turns in advance.  First, find all the legal moves.
    moves = legal_moves(player, board)
    
    # If player has no legal moves, then either:
    if not moves:
        # the game is over, so the best achievable score is victory or defeat;
        if not any_legal_move(opponent(player), board):
            return final_value(player, board), None
        # or we have to pass this turn, so just find the value of this board.
        return value(board), None
    
    # When there are multiple legal moves available, choose the best one by
    # maximizing the value of the resulting boards.
    return max((value(make_move(m, player, list(board))), m) for m in moves)

# Values for endgame boards are big constants.
MAX_VALUE = sum(map(abs, SQUARE_WEIGHTS))
MIN_VALUE = -MAX_VALUE

def final_value(player, board):
    """The game is over--find the value of this board to player."""
    diff = score(player, board)
    if diff < 0:
        return MIN_VALUE
    elif diff > 0:
        return MAX_VALUE
    return diff

def minimax_searcher(depth, evaluate):
    """
    Construct a strategy that uses `minimax` with the specified leaf board
    evaluation function.
    """
    def strategy(player, board):
        return minimax(player, board, depth, evaluate)[1]
    return strategy

### Alpha-Beta search

"""
http://en.wikipedia.org/wiki/Alpha-beta_pruning

Minimax is very effective, but it does too much work: it evaluates many search
trees that should be ignored.
Consider what happens when minimax is evaluating two moves, M1 and M2, on one
level of a search tree.  Suppose minimax determines that M1 can result in a
score of S.  While evaluating M2, if minimax finds a move in its subtree that
could result in a better score than S, the algorithm should immediately quit
evaluating M2: the opponent will force us to play M1 to avoid the higher score
resulting from M1, so we shouldn't waste time determining just how much better
M2 is than M1.

We need to keep track of two values:
 
- alpha: the maximum score achievable by any of the moves we have encountered.
- beta: the score that the opponent can keep us under by playing other moves.

When the algorithm begins, alpha is the smallest value and beta is the largest
value.  During evaluation, if we find a move that causes `alpha >= beta`, then
we can quit searching this subtree since the opponent can prevent us from
playing it.
"""

def alphabeta(player, board, alpha, beta, depth, evaluate):
    """
    Find the best legal move for player, searching to the specified depth.  Like
    minimax, but uses the bounds alpha and beta to prune branches.
    """
    if depth == 0:
        return evaluate(player, board), None

    def value(board, alpha, beta):
        # Like in `minimax`, the value of a board is the opposite of its value
        # to the opponent.  We pass in `-beta` and `-alpha` as the alpha and
        # beta values, respectively, for the opponent, since `alpha` represents
        # the best score we know we can achieve and is therefore the worst score
        # achievable by the opponent.  Similarly, `beta` is the worst score that
        # our opponent can hold us to, so it is the best score that they can
        # achieve.
        return -alphabeta(opponent(player), board, -beta, -alpha, depth-1, evaluate)[0]
    
    moves = legal_moves(player, board)
    if not moves:
        if not any_legal_move(opponent(player), board):
            return final_value(player, board), None
        return value(board, alpha, beta), None
    
    best_move = moves[0]
    for move in moves:
        if alpha >= beta:
            # If one of the legal moves leads to a better score than beta, then
            # the opponent will avoid this branch, so we can quit looking.
            break
        val = value(make_move(move, player, list(board)), alpha, beta)
        if val > alpha:
            # If one of the moves leads to a better score than the current best
            # achievable score, then replace it with this one.
            alpha = val
            best_move = move
    return alpha, best_move

def alphabeta_searcher(depth, evaluate):
    def strategy(player, board):
        return alphabeta(player, board, MIN_VALUE, MAX_VALUE, depth, evaluate)[1]
    return strategy
