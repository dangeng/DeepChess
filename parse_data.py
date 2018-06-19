import numpy as np
import chess.pgn

def get_bitboard(board):
    '''
    params
    ------

    board : chess.pgn board object
        board to get state from

    returns
    -------

    bitboard representation of the state of the game
    64 * 6 + 5 dim binary numpy vector
    64 squares, 6 pieces, '1' indicates the piece is at a square
    5 extra dimensions for castling rights queenside/kingside and whose turn

    '''

    bitboard = np.zeros(64*6*2+5)

    piece_idx = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}

    for i in range(64):
        if board.piece_at(i):
            color = int(board.piece_at(i).color) + 1
            bitboard[(piece_idx[board.piece_at(i).symbol().lower()] + i * 6) * color] = 1

    bitboard[-1] = int(board.turn)
    bitboard[-2] = int(board.has_kingside_castling_rights(True))
    bitboard[-3] = int(board.has_kingside_castling_rights(False))
    bitboard[-4] = int(board.has_queenside_castling_rights(True))
    bitboard[-5] = int(board.has_queenside_castling_rights(False))

    return bitboard

def get_result(game):
    result = game.headers['Result']
    result = result.split('-')
    if result[0] == '1':
        return 1
    elif result[0] == '0':
        return -1
    else:
        return 0

games = open('games.pgn')
bitboards = []
labels = []
num_games = 0

for i in range(845910):
    if num_games % 10 == 0:
        print(num_games)

    num_games += 1

    game = chess.pgn.read_game(games)

    result = get_result(game)

    board = game.board()

    for move in game.main_line():
        board.push(move)
        bitboard = get_bitboard(board)
        bitboards.append(bitboard)
        labels.append(result)

bitboards = np.array(bitboards)

