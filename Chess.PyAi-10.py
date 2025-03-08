import tkinter as tk
from tkinter import messagebox, simpledialog, colorchooser
import copy, time, json, os, threading, random

# ==================================
# إعدادات عامة والذاكرة
# ==================================
SQUARE_SIZE = 60
BOARD_SIZE = SQUARE_SIZE * 8

ai_memory = {}
MEMORY_FILE = "ai_memory.json"
CUSTOM_CONFIG_FILE = "custom_board.json"

def memory_key_to_str(key):
    return json.dumps(key)

def load_ai_memory():
    global ai_memory
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                ai_memory = json.load(f)
        except Exception as e:
            ai_memory = {}
    else:
        ai_memory = {}

def save_ai_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump(ai_memory, f)

load_ai_memory()

def load_custom_config():
    if os.path.exists(CUSTOM_CONFIG_FILE):
        try:
            with open(CUSTOM_CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            return None
    return None

def save_custom_config(config):
    with open(CUSTOM_CONFIG_FILE, "w") as f:
        json.dump(config, f)

# ==================================
# إعدادات Zobrist Hashing للوحة
# ==================================
ZOBRIST_TABLE = {}
ZOBRIST_TURN = None

def init_zobrist():
    global ZOBRIST_TABLE, ZOBRIST_TURN
    pieces = ['wP','wN','wB','wR','wQ','wK','bP','bN','bB','bR','bQ','bK']
    for r in range(8):
        for c in range(8):
            for p in pieces:
                ZOBRIST_TABLE[(r, c, p)] = random.getrandbits(64)
    ZOBRIST_TURN = random.getrandbits(64)

init_zobrist()

def compute_zobrist_hash(board, turn):
    h = 0
    for r in range(8):
        for c in range(8):
            piece = board.get_piece(r, c)
            if piece:
                key = piece.color + piece.kind.upper()
                h ^= ZOBRIST_TABLE.get((r, c, key), 0)
    if turn == 'w':
        h ^= ZOBRIST_TURN
    return h

# ==================================
# Piece-Square Tables لتحسين تقييم القطع (الافتتاح)
# ==================================
piece_square_tables = {
    'P': [
        [0,0,0,0,0,0,0,0],
        [5,10,10,-20,-20,10,10,5],
        [5,-5,-10,0,0,-10,-5,5],
        [0,0,0,20,20,0,0,0],
        [5,5,10,25,25,10,5,5],
        [10,10,20,30,30,20,10,10],
        [50,50,50,50,50,50,50,50],
        [0,0,0,0,0,0,0,0]
    ],
    'N': [
        [-50,-40,-30,-30,-30,-30,-40,-50],
        [-40,-20,0,0,0,0,-20,-40],
        [-30,0,10,15,15,10,0,-30],
        [-30,5,15,20,20,15,5,-30],
        [-30,0,15,20,20,15,0,-30],
        [-30,5,10,15,15,10,5,-30],
        [-40,-20,0,5,5,0,-20,-40],
        [-50,-40,-30,-30,-30,-30,-40,-50]
    ],
    'B': [
        [-20,-10,-10,-10,-10,-10,-10,-20],
        [-10,0,0,0,0,0,0,-10],
        [-10,0,5,10,10,5,0,-10],
        [-10,5,5,10,10,5,5,-10],
        [-10,0,10,10,10,10,0,-10],
        [-10,10,10,10,10,10,10,-10],
        [-10,5,0,0,0,0,5,-10],
        [-20,-10,-10,-10,-10,-10,-10,-20]
    ],
    'R': [
        [0,0,0,0,0,0,0,0],
        [5,10,10,10,10,10,10,5],
        [-5,0,0,0,0,0,0,-5],
        [-5,0,0,0,0,0,0,-5],
        [-5,0,0,0,0,0,0,-5],
        [-5,0,0,0,0,0,0,-5],
        [-5,0,0,0,0,0,0,-5],
        [0,0,0,5,5,0,0,0]
    ],
    'Q': [
        [-20,-10,-10,-5,-5,-10,-10,-20],
        [-10,0,0,0,0,0,0,-10],
        [-10,0,5,5,5,5,0,-10],
        [-5,0,5,5,5,5,0,-5],
        [0,0,5,5,5,5,0,-5],
        [-10,5,5,5,5,5,0,-10],
        [-10,0,5,0,0,0,0,-10],
        [-20,-10,-10,-5,-5,-10,-10,-20]
    ],
    'K': [
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-20,-30,-30,-40,-40,-30,-30,-20],
        [-10,-20,-20,-20,-20,-20,-20,-10],
        [20,20,0,0,0,0,20,20],
        [20,30,10,0,0,10,30,20]
    ]
}

# ==================================
# Piece-Square Table لنهاية اللعبة للملك (تشجيع التمركز)
# ==================================
piece_square_tables_endgame = {
    'K': [
        [-50,-30,-30,-30,-30,-30,-30,-50],
        [-30,-10,-10,-10,-10,-10,-10,-30],
        [-30,-10,20,20,20,20,-10,-30],
        [-30,-10,20,40,40,20,-10,-30],
        [-30,-10,20,40,40,20,-10,-30],
        [-30,-10,20,20,20,20,-10,-30],
        [-30,-20,-10,-10,-10,-10,-20,-30],
        [-50,-40,-30,-20,-20,-30,-40,-50]
    ]
}

def get_piece_square_bonus(piece, r, c, endgame=False):
    if piece is None or piece.kind.upper() not in piece_square_tables:
        return 0
    if piece.kind.upper() == 'K' and endgame:
         table = piece_square_tables_endgame['K']
         return table[7 - r][c] if piece.color == 'w' else table[r][c]
    else:
         table = piece_square_tables[piece.kind.upper()]
         return table[7 - r][c] if piece.color == 'w' else table[r][c]

# ==================================
# تعريف القطع ولوحة الشطرنج (الكود الأساسي)
# ==================================
PIECE_UNICODE = {
    ('w', 'K'): '♔', ('w', 'Q'): '♕', ('w', 'R'): '♖',
    ('w', 'B'): '♗', ('w', 'N'): '♘', ('w', 'P'): '♙',
    ('b', 'K'): '♚', ('b', 'Q'): '♛', ('b', 'R'): '♜',
    ('b', 'B'): '♝', ('b', 'N'): '♞', ('b', 'P'): '♟',
}

piece_values = {
    'P': 100,
    'N': 320,
    'B': 330,
    'R': 500,
    'Q': 900,
    'K': 10000
}

def enemy(color):
    return 'b' if color == 'w' else 'w'

class Piece:
    def __init__(self, color, kind):
        self.color = color  # 'w' أو 'b'
        self.kind = kind    # 'P', 'N', 'B', 'R', 'Q', 'K'
        self.moved = False
    def copy(self):
        new_piece = Piece(self.color, self.kind)
        new_piece.moved = self.moved
        return new_piece

class Board:
    def __init__(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.en_passant_target = None
        self.move_history = []
        self.halfmove_clock = 0
        self.setup_board()
    def setup_board(self):
        for col in range(8):
            self.board[1][col] = Piece('b', 'P')
            self.board[6][col] = Piece('w', 'P')
        order = ['R','N','B','Q','K','B','N','R']
        for col, kind in enumerate(order):
            self.board[0][col] = Piece('b', kind)
            self.board[7][col] = Piece('w', kind)
    def copy(self):
        new_board = Board.__new__(Board)
        new_board.board = [[self.board[r][c].copy() if self.board[r][c] else None for c in range(8)] for r in range(8)]
        new_board.en_passant_target = self.en_passant_target
        new_board.move_history = self.move_history.copy()
        new_board.halfmove_clock = self.halfmove_clock
        return new_board
    def in_bounds(self, r, c):
        return 0 <= r < 8 and 0 <= c < 8
    def get_piece(self, r, c):
        return self.board[r][c] if self.in_bounds(r, c) else None
    def move_piece(self, sr, sc, dr, dc, play_sound=False):
        piece = self.board[sr][sc]
        if piece is None:
            return  # تفادي الخطأ إذا كانت الخانة فارغة
        target = self.board[dr][dc]
        is_capture = (target is not None)
        if piece.kind == 'P' and (dr, dc) == self.en_passant_target and sc != dc and target is None:
            is_capture = True
            if piece.color == 'w':
                self.board[dr+1][dc] = None
            else:
                self.board[dr-1][dc] = None
        self.board[dr][dc] = piece
        self.board[sr][sc] = None
        piece.moved = True
        if piece.kind == 'P' or is_capture:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1
        if piece.kind == 'P' and abs(dr - sr) == 2:
            self.en_passant_target = ((sr + dr)//2, sc)
        else:
            self.en_passant_target = None
        if piece.kind == 'P':
            if (piece.color == 'w' and dr == 0) or (piece.color == 'b' and dr == 7):
                piece.kind = 'Q'
        if piece.kind == 'K' and abs(dc - sc) == 2:
            if dc == 6:
                rook = self.get_piece(sr,7)
                self.board[sr][5] = rook
                self.board[sr][7] = None
            elif dc == 2:
                rook = self.get_piece(sr,0)
                self.board[sr][3] = rook
                self.board[sr][0] = None
        self.move_history.append(((sr, sc),(dr, dc)))
    def is_square_attacked(self, r, c, attacker_color, check_castling=False):
        for i in range(8):
            for j in range(8):
                p = self.get_piece(i,j)
                if p and p.color == attacker_color:
                    moves = self.get_pseudo_legal_moves(i,j,ignore_checks=True, check_castling=check_castling)
                    if (r,c) in moves:
                        return True
        return False
    def is_in_check(self, color, check_castling=True):
        king_pos = None
        for r in range(8):
            for c in range(8):
                p = self.get_piece(r,c)
                if p and p.color == color and p.kind == 'K':
                    king_pos = (r,c)
                    break
            if king_pos:
                break
        if not king_pos:
            return True
        return self.is_square_attacked(king_pos[0], king_pos[1], enemy(color), check_castling=check_castling)
    def get_pseudo_legal_moves(self, r, c, ignore_checks=False, check_castling=True):
        moves = []
        piece = self.get_piece(r,c)
        if not piece:
            return moves
        if piece.kind == 'P':
            direction = -1 if piece.color=='w' else 1
            if self.in_bounds(r+direction, c) and self.get_piece(r+direction, c) is None:
                moves.append((r+direction,c))
                if not piece.moved and self.in_bounds(r+2*direction, c) and self.get_piece(r+2*direction, c) is None:
                    moves.append((r+2*direction, c))
            for dc in [-1,1]:
                nr, nc = r+direction, c+dc
                if self.in_bounds(nr, nc):
                    target = self.get_piece(nr,nc)
                    if target and target.color != piece.color:
                        moves.append((nr,nc))
                    if (nr,nc) == self.en_passant_target:
                        moves.append((nr,nc))
        elif piece.kind == 'N':
            for dr, dc in [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]:
                nr, nc = r+dr, c+dc
                if self.in_bounds(nr,nc):
                    target = self.get_piece(nr,nc)
                    if target is None or target.color != piece.color:
                        moves.append((nr,nc))
        if piece.kind in ['B','Q']:
            for dr, dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = r+dr, c+dc
                while self.in_bounds(nr,nc):
                    target = self.get_piece(nr,nc)
                    if target is None:
                        moves.append((nr,nc))
                    else:
                        if target.color != piece.color:
                            moves.append((nr,nc))
                        break
                    nr += dr
                    nc += dc
        if piece.kind in ['R','Q']:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                while self.in_bounds(nr,nc):
                    target = self.get_piece(nr,nc)
                    if target is None:
                        moves.append((nr,nc))
                    else:
                        if target.color != piece.color:
                            moves.append((nr,nc))
                        break
                    nr += dr
                    nc += dc
        if piece.kind == 'K':
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if dr==0 and dc==0:
                        continue
                    nr, nc = r+dr, c+dc
                    if self.in_bounds(nr,nc):
                        target = self.get_piece(nr,nc)
                        if target is None or target.color != piece.color:
                            moves.append((nr,nc))
            if not piece.moved and check_castling and not self.is_in_check(piece.color, check_castling=False):
                rook = self.get_piece(r,7)
                if rook and rook.kind=='R' and not rook.moved:
                    if self.get_piece(r,5) is None and self.get_piece(r,6) is None:
                        if not self.is_square_attacked(r,5,enemy(piece.color), check_castling=False) and not self.is_square_attacked(r,6,enemy(piece.color), check_castling=False):
                            moves.append((r,6))
                rook = self.get_piece(r,0)
                if rook and rook.kind=='R' and not rook.moved:
                    if self.get_piece(r,1) is None and self.get_piece(r,2) is None and self.get_piece(r,3) is None:
                        if not self.is_square_attacked(r,2,enemy(piece.color), check_castling=False) and not self.is_square_attacked(r,3,enemy(piece.color), check_castling=False):
                            moves.append((r,2))
        return moves
    def get_valid_moves(self, r, c):
        valid_moves = []
        pseudo_moves = self.get_pseudo_legal_moves(r,c)
        for move in pseudo_moves:
            new_board = self.copy()
            new_board.move_piece(r,c,move[0],move[1], play_sound=False)
            if not new_board.is_in_check(self.get_piece(r,c).color):
                valid_moves.append(move)
        return valid_moves

def get_all_valid_moves(board, color):
    moves = []
    for r in range(8):
        for c in range(8):
            piece = board.get_piece(r,c)
            if piece and piece.color==color:
                valid = board.get_valid_moves(r,c)
                for move in valid:
                    moves.append(((r,c), move))
    return moves

# ==================================
# تقييم اللوحة باستخدام Bitboards
# ==================================
def rc_to_sq(r,c):
    return (7-r)*8+c

def board_to_bitboards(board_obj):
    bitboards = {('w','P'):0,('w','N'):0,('w','B'):0,('w','R'):0,('w','Q'):0,('w','K'):0,
                 ('b','P'):0,('b','N'):0,('b','B'):0,('b','R'):0,('b','Q'):0,('b','K'):0}
    for r in range(8):
        for c in range(8):
            piece = board_obj.get_piece(r,c)
            if piece:
                sq = rc_to_sq(r,c)
                key = (piece.color, piece.kind.upper())
                bitboards[key] |= (1 << sq)
    return bitboards

def popcount(x):
    return bin(x).count("1")

def evaluate_board_bitboards(bitboards, ai_color):
    score = 0
    endgame = (bitboards.get(('w','Q'),0)==0 and bitboards.get(('b','Q'),0)==0)
    for key, bb in bitboards.items():
        color, kind = key
        piece_val = piece_values.get(kind,0)
        temp = bb
        while temp:
            lsb = temp & -temp
            sq = lsb.bit_length()-1
            r = 7 - (sq//8)
            c = sq % 8
            bonus = get_piece_square_bonus(Piece(color,kind), r, c, endgame if kind.upper()=='K' else False)
            if color==ai_color:
                score += piece_val + bonus
            else:
                score -= piece_val + bonus
            temp &= temp - 1
    return score

def board_state_terminal(board, ai_color):
    if board.halfmove_clock>=100:
        return True
    return (not get_all_valid_moves(board, ai_color) or not get_all_valid_moves(board, enemy(ai_color)))

def is_checkmate(board, color):
    moves = get_all_valid_moves(board, color)
    return (not moves) and board.is_in_check(color)

def is_stalemate(board, color):
    moves = get_all_valid_moves(board, color)
    return (not moves) and (not board.is_in_check(color))

class ChessAI:
    def __init__(self, color, max_depth=4, time_limit=5):
        self.color = color
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.transposition = {}
        self.start_time = 0
        self.killer_moves = {}
        self.opening_book = {}
    def board_to_key(self, board, turn):
        return compute_zobrist_hash(board, turn)
    def opening_book_move(self, board):
        history = board.move_history
        if self.color=='w':
            if len(history)==0:
                return ((6,4),(4,4))
            if len(history)==1:
                return None
            if len(history)==2:
                black_move = history[1]
                if black_move==((1,4),(3,4)):
                    return ((7,6),(5,5))
                if black_move==((1,4),(2,4)):
                    return ((6,3),(4,3))
                if black_move==((1,2),(3,2)):
                    return ((6,3),(4,3))
            if len(history)==4:
                if history[0]==((6,4),(4,4)) and history[1]==((1,4),(3,4)) and history[2]==((7,6),(5,5)):
                    black_move = history[3]
                    if black_move==((0,1),(2,2)):
                        return random.choice([((7,5),(4,2)), ((7,5),(3,1))])
                if history[0]==((6,4),(4,4)) and history[1]==((1,4),(2,4)) and history[2]==((6,3),(4,3)):
                    black_move = history[3]
                    if black_move==((1,3),(3,3)):
                        return ((4,4),(3,3))
                if history[0]==((6,4),(4,4)) and history[1]==((1,2),(3,2)) and history[2]==((6,3),(4,3)):
                    black_move = history[3]
                    if black_move==((1,3),(3,3)):
                        return random.choice([((4,4),(3,4)), ((4,4),(3,3))])
        else:
            if len(history)==1:
                white_move = history[0]
                if white_move==((6,4),(4,4)):
                    return ((1,4),(2,4))
                elif white_move==((6,3),(4,3)):
                    return ((1,3),(2,3))
                elif white_move==((6,2),(4,2)):
                    return ((1,4),(2,4))
        return None
    def is_bad_queen_sacrifice(self, board, move, check_depth=3):
        sr, sc = move[0]
        piece = board.get_piece(sr, sc)
        if not piece or piece.kind.upper()!='Q':
            return False
        new_board = board.copy()
        dr, dc = move[1]
        new_board.move_piece(sr, sc, dr, dc, play_sound=False)
        queen_exists = any(new_board.get_piece(r,c) and new_board.get_piece(r,c).color==piece.color and new_board.get_piece(r,c).kind.upper()=='Q'
                           for r in range(8) for c in range(8))
        if queen_exists:
            return False
        score, _ = self.minimax(new_board, check_depth, -float('inf'), float('inf'), maximizing=False)
        return score < -500
    def choose_move(self, board):
        book_move = self.opening_book_move(board)
        if book_move is not None:
            return book_move
        self.transposition.clear()
        self.killer_moves.clear()
        best_move = None
        best_score = -float('inf')
        self.start_time = time.time()
        current_best = None
        for depth in range(1, self.max_depth+1):
            if time.time()-self.start_time > self.time_limit:
                break
            score, move = self.minimax(board, depth, -float('inf'), float('inf'), True)
            if move is not None:
                current_best = move
                best_score = score
        if current_best is not None:
            return current_best
        valid = get_all_valid_moves(board, self.color)
        return random.choice(valid) if valid else None
    def quiescence(self, board, alpha, beta, maximizing):
        bb = board_to_bitboards(board)
        stand_pat = evaluate_board_bitboards(bb, self.color)
        if maximizing:
            if stand_pat>=beta:
                return beta, None
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat<=alpha:
                return alpha, None
            beta = min(beta, stand_pat)
        current_color = self.color if maximizing else enemy(self.color)
        capture_moves = []
        for r in range(8):
            for c in range(8):
                piece = board.get_piece(r,c)
                if piece and piece.color==current_color:
                    for move in board.get_valid_moves(r,c):
                        dr, dc = move
                        target = board.get_piece(dr,dc)
                        if target is not None or (piece.kind=='P' and (dr,dc)==board.en_passant_target):
                            capture_moves.append(((r,c), move))
        if not capture_moves:
            return stand_pat, None
        best_move = None
        if maximizing:
            for move in capture_moves:
                new_board = board.copy()
                sr, sc = move[0]
                dr, dc = move[1]
                new_board.move_piece(sr, sc, dr, dc, play_sound=False)
                score, _ = self.quiescence(new_board, alpha, beta, False)
                if score>alpha:
                    alpha = score
                    best_move = move
                if alpha>=beta:
                    break
            return alpha, best_move
        else:
            for move in capture_moves:
                new_board = board.copy()
                sr, sc = move[0]
                dr, dc = move[1]
                new_board.move_piece(sr, sc, dr, dc, play_sound=False)
                score, _ = self.quiescence(new_board, alpha, beta, True)
                if score<beta:
                    beta = score
                    best_move = move
                if beta<=alpha:
                    break
            return beta, best_move
    def minimax(self, board, depth, alpha, beta, maximizing):
        current_turn = self.color if maximizing else enemy(self.color)
        key = self.board_to_key(board, current_turn)
        if key in self.transposition:
            stored_depth, stored_score, stored_move = self.transposition[key]
            if stored_depth>=depth:
                return stored_score, stored_move
        if time.time()-self.start_time > self.time_limit:
            bb = board_to_bitboards(board)
            return evaluate_board_bitboards(bb, self.color), None
        if board_state_terminal(board, self.color):
            if is_checkmate(board, enemy(self.color)):
                return 100000-(self.max_depth-depth), None
            if is_checkmate(board, self.color):
                return -100000+(self.max_depth-depth), None
            if is_stalemate(board, enemy(self.color)) or is_stalemate(board, self.color):
                return 0, None
        if depth==0:
            return self.quiescence(board, alpha, beta, maximizing)
        moves = get_all_valid_moves(board, current_turn)
        if depth in self.killer_moves:
            killer = self.killer_moves[depth]
            for km in killer:
                if km in moves:
                    moves.remove(km)
                    moves.insert(0, km)
        moves.sort(key=lambda m: evaluate_board_bitboards(board_to_bitboards(make_temp_board(board, m, current_turn)), self.color), reverse=maximizing)
        filtered_moves = []
        if maximizing and current_turn==self.color:
            for move in moves:
                sr, sc = move[0]
                piece = board.get_piece(sr,sc)
                if piece and piece.kind.upper()=='Q' and self.is_bad_queen_sacrifice(board, move):
                    continue
                filtered_moves.append(move)
            if not filtered_moves:
                filtered_moves = moves
        else:
            filtered_moves = moves
        best_move = None
        if maximizing:
            best_score = -float('inf')
            for move in filtered_moves:
                new_board = board.copy()
                sr, sc = move[0]
                dr, dc = move[1]
                new_board.move_piece(sr, sc, dr, dc, play_sound=False)
                score, _ = self.minimax(new_board, depth-1, alpha, beta, False)
                if score>best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
                if beta<=alpha:
                    self.killer_moves.setdefault(depth, []).append(move)
                    break
            self.transposition[key]=(depth, best_score, best_move)
            if len(self.transposition)>10000:
                self.transposition.clear()
            return best_score, best_move
        else:
            best_score = float('inf')
            for move in filtered_moves:
                new_board = board.copy()
                sr, sc = move[0]
                dr, dc = move[1]
                new_board.move_piece(sr, sc, dr, dc, play_sound=False)
                score, _ = self.minimax(new_board, depth-1, alpha, beta, True)
                if score<best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
                if beta<=alpha:
                    self.killer_moves.setdefault(depth, []).append(move)
                    break
            self.transposition[key]=(depth, best_score, best_move)
            if len(self.transposition)>10000:
                self.transposition.clear()
            return best_score, best_move

def make_temp_board(board, move, color):
    new_board = board.copy()
    sr, sc = move[0]
    dr, dc = move[1]
    new_board.move_piece(sr, sc, dr, dc, play_sound=False)
    return new_board

def get_board_key(board):
    board_tuple = []
    for r in range(8):
        row = []
        for c in range(8):
            piece = board.get_piece(r,c)
            row.append(piece.color+piece.kind.upper() if piece else '.')
        board_tuple.append(tuple(row))
    return tuple(board_tuple)

def simulate_ai_vs_ai_game(num_games=100, depth=2, time_limit=1):
    for _ in range(num_games):
        board = Board()
        turn = "w"
        ai_white = ChessAI("w", max_depth=depth, time_limit=time_limit)
        ai_black = ChessAI("b", max_depth=depth, time_limit=time_limit)
        game_moves = []
        while True:
            if board.halfmove_clock>=100 or not get_all_valid_moves(board, turn):
                break
            ai_player = ai_white if turn=="w" else ai_black
            move = ai_player.choose_move(board)
            if move is None:
                break
            mem_key = memory_key_to_str((get_board_key(board), move))
            game_moves.append(mem_key)
            sr, sc = move[0]
            dr, dc = move[1]
            board.move_piece(sr, sc, dr, dc, play_sound=False)
            turn = enemy(turn)
        winner = enemy(turn) if board.is_in_check(turn) else None
        for i, key in enumerate(game_moves):
            delta = 0 if winner is None else (30 if ('w' if i%2==0 else 'b')==winner else -30)
            ai_memory[key] = ai_memory.get(key,0)+delta
        save_ai_memory()


# واجهة المستخدم باستخدام Tkinter (تحديث 10)
# ==================================
class ChessTkinterGame:
    def __init__(self, root):
        self.root = root
        self.language = "EN"
        self.light_color = "#eeeed2"
        self.dark_color = "#769656"
        self.highlight_color = "#FF0000"
        self.piece_color_white = "#000000"
        self.piece_color_black = "#000000"
        self.ai_depth = 4
        self.ai_time_limit = 5
        self.show_moves = True
        self.move_highlight_shape = "square"
        self.game_mode = tk.StringVar(value="HvA")
        self.human_color = "w"
        self.custom_board_config = load_custom_config()
        self.set_titles()
        self.root.title(self.title_text)
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(pady=5)
        self.mode_label = tk.Label(self.top_frame, text=self.select_mode_text)
        self.mode_label.grid(row=0, column=0, padx=5)
        tk.Radiobutton(self.top_frame, text=self.hva_text, variable=self.game_mode, value="HvA").grid(row=0, column=1, padx=5)
        tk.Radiobutton(self.top_frame, text=self.hva_black_text, variable=self.game_mode, value="HvA_Black").grid(row=0, column=2, padx=5)
        tk.Radiobutton(self.top_frame, text=self.hvh_text, variable=self.game_mode, value="HvH").grid(row=0, column=3, padx=5)
        tk.Radiobutton(self.top_frame, text=self.ava_text, variable=self.game_mode, value="AvA").grid(row=0, column=4, padx=5)
        tk.Radiobutton(self.top_frame, text="Custom", variable=self.game_mode, value="Custom").grid(row=0, column=5, padx=5)
        self.run_button = tk.Button(self.top_frame, text="Run", command=self.run_custom_game)
        self.run_button.grid(row=0, column=6, padx=5)
        self.new_game_button = tk.Button(self.top_frame, text=self.new_game_text, command=self.new_game)
        self.new_game_button.grid(row=1, column=0, padx=5, pady=5)
        self.custom_button = tk.Button(self.top_frame, text="Custom Chess", command=self.open_custom_chess_editor)
        self.custom_button.grid(row=1, column=1, padx=5, pady=5)
        self.undo_button = tk.Button(self.top_frame, text=self.undo_text, command=self.undo_move)
        self.undo_button.grid(row=1, column=2, padx=5, pady=5)
        self.train_button = tk.Button(self.top_frame, text="Train AI", command=self.train_ai)
        self.train_button.grid(row=1, column=3, padx=5, pady=5)
        self.lang_button = tk.Button(self.top_frame, text=self.toggle_lang_text, command=self.toggle_language)
        self.lang_button.grid(row=1, column=4, padx=5, pady=5)
        self.settings_button = tk.Button(self.top_frame, text=self.settings_text, command=self.open_settings_window)
        self.settings_button.grid(row=1, column=5, padx=5, pady=5)
        self.status_label = tk.Label(self.top_frame, text=self.wait_text, font=('Arial', 14))
        self.status_label.grid(row=2, column=0, columnspan=7, padx=5)
        self.canvas = tk.Canvas(root, width=BOARD_SIZE, height=BOARD_SIZE)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.board = None
        self.turn = None
        self.selected = None
        self.valid_moves = []
        self.board_history = []
        self.last_move_dest = None
        self.ai_game_moves = []
        self.ai = None
        self.ai_white = None
        self.ai_black = None
        self.new_game()
        self.game_loop()
    def set_titles(self):
        if self.language=="EN":
            self.title_text = "Chess Game with AI v10 (Custom Options)"
            self.select_mode_text = "Select game mode:"
            self.hva_text = "Human vs AI (You are White)"
            self.hva_black_text = "Human vs AI (You are Black)"
            self.hvh_text = "Human vs Human"
            self.ava_text = "AI vs AI"
            self.new_game_text = "New Game"
            self.undo_text = "Undo"
            self.toggle_lang_text = "عربي"
            self.wait_text = "Waiting for game to start..."
            self.your_turn_text = "Your turn ("
            self.ai_turn_text = "AI turn ("
            self.turn_text = "Turn: "
            self.checkmate_text = "Checkmate! "
            self.win_white_text = "White wins."
            self.win_black_text = "Black wins."
            self.draw_text = "Draw!"
            self.settings_text = "Settings"
        else:
            self.title_text = "لعبة شطرنج مع الذكاء الاصطناعي v10 (خيارات مخصصة)"
            self.select_mode_text = "اختر نمط اللعبة:"
            self.hva_text = "بشر vs AI (أنت الأبيض)"
            self.hva_black_text = "بشر vs AI (أنت الأسود)"
            self.hvh_text = "بشر ضد بشر"
            self.ava_text = "AI ضد AI"
            self.new_game_text = "لعبة جديدة"
            self.undo_text = "تراجع"
            self.toggle_lang_text = "English"
            self.wait_text = "انتظار بدء اللعبة..."
            self.your_turn_text = "دورك ("
            self.ai_turn_text = "دور الذكاء الاصطناعي ("
            self.turn_text = "الدور: "
            self.checkmate_text = "كش ملك! "
            self.win_white_text = "فاز الأبيض."
            self.win_black_text = "فاز الأسود."
            self.draw_text = "تعادل!"
            self.settings_text = "الإعدادات"
        if hasattr(self, 'settings_button'):
            self.settings_button.config(text=self.settings_text)
    def toggle_language(self):
        self.language = "AR" if self.language=="EN" else "EN"
        self.set_titles()
        self.root.title(self.title_text)
        self.mode_label.config(text=self.select_mode_text)
        self.new_game_button.config(text=self.new_game_text)
        self.undo_button.config(text=self.undo_text)
        self.lang_button.config(text=self.toggle_lang_text)
        self.status_label.config(text=self.wait_text)
    def open_settings_window(self):
        settings_win = tk.Toplevel(self.root)
        settings_win.title("Settings" if self.language=="EN" else "الإعدادات")
        board_frame = tk.LabelFrame(settings_win, text="Board Settings" if self.language=="EN" else "إعدادات الرقعة")
        board_frame.pack(padx=10, pady=10, fill="both", expand=True)
        tk.Label(board_frame, text="Light Color:" if self.language=="EN" else "لون المربع الفاتح:").grid(row=0, column=0, padx=5, pady=5)
        light_entry = tk.Entry(board_frame)
        light_entry.insert(0, self.light_color)
        light_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(board_frame, text="Dark Color:" if self.language=="EN" else "لون المربع الغامق:").grid(row=1, column=0, padx=5, pady=5)
        dark_entry = tk.Entry(board_frame)
        dark_entry.insert(0, self.dark_color)
        dark_entry.grid(row=1, column=1, padx=5, pady=5)
        piece_frame = tk.LabelFrame(settings_win, text="Piece Colors" if self.language=="EN" else "ألوان القطع")
        piece_frame.pack(padx=10, pady=10, fill="both", expand=True)
        tk.Label(piece_frame, text="White Pieces:" if self.language=="EN" else "قطع الأبيض:").grid(row=0, column=0, padx=5, pady=5)
        piece_white_entry = tk.Entry(piece_frame)
        piece_white_entry.insert(0, self.piece_color_white)
        piece_white_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(piece_frame, text="Black Pieces:" if self.language=="EN" else "قطع الأسود:").grid(row=1, column=0, padx=5, pady=5)
        piece_black_entry = tk.Entry(piece_frame)
        piece_black_entry.insert(0, self.piece_color_black)
        piece_black_entry.grid(row=1, column=1, padx=5, pady=5)
        move_frame = tk.LabelFrame(settings_win, text="Move Highlight Options" if self.language=="EN" else "خيارات تمييز النقلات")
        move_frame.pack(padx=10, pady=10, fill="both", expand=True)
        show_moves_var = tk.BooleanVar(value=self.show_moves)
        tk.Checkbutton(move_frame, text="Show possible moves" if self.language=="EN" else "إظهار النقلات الممكنة", variable=show_moves_var).grid(row=0, column=0, padx=5, pady=5)
        tk.Label(move_frame, text="Highlight Shape:" if self.language=="EN" else "شكل التمييز:").grid(row=1, column=0, padx=5, pady=5)
        shape_var = tk.StringVar(value=self.move_highlight_shape)
        tk.OptionMenu(move_frame, shape_var, "square", "circle").grid(row=1, column=1, padx=5, pady=5)
        ai_frame = tk.LabelFrame(settings_win, text="AI Settings" if self.language=="EN" else "إعدادات الذكاء الاصطناعي")
        ai_frame.pack(padx=10, pady=10, fill="both", expand=True)
        tk.Label(ai_frame, text="Search Depth:" if self.language=="EN" else "عمق البحث:").grid(row=0, column=0, padx=5, pady=5)
        depth_entry = tk.Entry(ai_frame)
        depth_entry.insert(0, str(self.ai_depth))
        depth_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(ai_frame, text="Time Limit (s):" if self.language=="EN" else "حد الوقت (ثواني):").grid(row=1, column=0, padx=5, pady=5)
        time_entry = tk.Entry(ai_frame)
        time_entry.insert(0, str(self.ai_time_limit))
        time_entry.grid(row=1, column=1, padx=5, pady=5)
        def save_settings():
            self.light_color = light_entry.get()
            self.dark_color = dark_entry.get()
            self.piece_color_white = piece_white_entry.get()
            self.piece_color_black = piece_black_entry.get()
            self.show_moves = show_moves_var.get()
            self.move_highlight_shape = shape_var.get()
            try:
                self.ai_depth = int(depth_entry.get())
            except:
                self.ai_depth = 4
            try:
                self.ai_time_limit = float(time_entry.get())
            except:
                self.ai_time_limit = 5
            self.new_game()
            settings_win.destroy()
        save_button = tk.Button(settings_win, text="Save" if self.language=="EN" else "حفظ", command=save_settings)
        save_button.pack(pady=10)
    def run_custom_game(self):
        if self.game_mode.get()=="Custom":
            self.ai_white = ChessAI("w", max_depth=self.ai_depth, time_limit=self.ai_time_limit)
            self.ai_black = ChessAI("b", max_depth=self.ai_depth, time_limit=self.ai_time_limit)
            messagebox.showinfo(self.title_text, "Custom game running!")
        else:
            messagebox.showinfo(self.title_text, "Run button is available only in Custom mode.")
    def open_custom_chess_editor(self):
        editor = tk.Toplevel(self.root)
        editor.title("Custom Chess Editor" if self.language=="EN" else "محرر الشطرنج المخصص")
        canvas_size = BOARD_SIZE
        edit_canvas = tk.Canvas(editor, width=canvas_size, height=canvas_size, bg="white")
        edit_canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)
        custom_board = [[None for _ in range(8)] for _ in range(8)]
        custom_board[7][4] = {"color": "w", "kind": "K"}
        custom_board[0][4] = {"color": "b", "kind": "K"}
        def draw_custom_board():
            edit_canvas.delete("all")
            for r in range(8):
                for c in range(8):
                    x1 = c * SQUARE_SIZE
                    y1 = r * SQUARE_SIZE
                    x2 = x1 + SQUARE_SIZE
                    y2 = y1 + SQUARE_SIZE
                    fill_color = self.light_color if (r+c)%2==0 else self.dark_color
                    edit_canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color)
                    cell = custom_board[r][c]
                    if cell:
                        piece_text = PIECE_UNICODE[(cell["color"], cell["kind"])]
                        color = self.piece_color_white if cell["color"]=='w' else self.piece_color_black
                        edit_canvas.create_text(x1+SQUARE_SIZE//2, y1+SQUARE_SIZE//2, text=piece_text, font=('Arial',32), fill=color)
        draw_custom_board()
        piece_choice = tk.StringVar(value="P")
        color_choice = tk.StringVar(value="w")
        tk.Label(editor, text="Piece:" if self.language=="EN" else "القطعة:").grid(row=1, column=0, padx=5, pady=5)
        tk.OptionMenu(editor, piece_choice, "P","N","B","R","Q").grid(row=1, column=1, padx=5, pady=5)
        tk.Label(editor, text="Color:" if self.language=="EN" else "اللون:").grid(row=1, column=2, padx=5, pady=5)
        tk.OptionMenu(editor, color_choice, "w","b").grid(row=1, column=3, padx=5, pady=5)
        def add_piece(event):
            col = event.x // SQUARE_SIZE
            row = event.y // SQUARE_SIZE
            if custom_board[row][col] and custom_board[row][col]["kind"]=="K":
                return
            custom_board[row][col] = {"color": color_choice.get(), "kind": piece_choice.get()}
            draw_custom_board()
        edit_canvas.bind("<Button-1>", add_piece)
        def delete_piece(event):
            col = event.x // SQUARE_SIZE
            row = event.y // SQUARE_SIZE
            if custom_board[row][col] and custom_board[row][col]["kind"]!="K":
                custom_board[row][col] = None
                draw_custom_board()
        edit_canvas.bind("<Button-3>", delete_piece)
        def clear_board():
            for r in range(8):
                for c in range(8):
                    if custom_board[r][c] and custom_board[r][c]["kind"]!="K":
                        custom_board[r][c] = None
            draw_custom_board()
        clear_button = tk.Button(editor, text="Clear" if self.language=="EN" else "مسح", command=clear_board)
        clear_button.grid(row=2, column=0, padx=5, pady=5)
        def save_custom():
            white_king = any(custom_board[r][c] and custom_board[r][c]["kind"]=="K" and custom_board[r][c]["color"]=="w" for r in range(8) for c in range(8))
            black_king = any(custom_board[r][c] and custom_board[r][c]["kind"]=="K" and custom_board[r][c]["color"]=="b" for r in range(8) for c in range(8))
            if not (white_king and black_king):
                messagebox.showerror("Error" if self.language=="EN" else "خطأ", "Both kings must be present!" if self.language=="EN" else "يجب أن يكون الملكان موجودين!")
                return
            self.custom_board_config = custom_board
            save_custom_config(custom_board)
            messagebox.showinfo("Saved" if self.language=="EN" else "تم الحفظ", "Custom board saved!" if self.language=="EN" else "تم حفظ التشكيلة المخصصة!")
            editor.destroy()
        save_button = tk.Button(editor, text="Save" if self.language=="EN" else "حفظ", command=save_custom)
        save_button.grid(row=2, column=1, padx=5, pady=5)
    def new_game(self):
        if self.game_mode.get()=="Custom" and self.custom_board_config:
            self.board = Board()
            for r in range(8):
                for c in range(8):
                    self.board.board[r][c] = None
            for r in range(8):
                for c in range(8):
                    cell = self.custom_board_config[r][c]
                    if cell:
                        self.board.board[r][c] = Piece(cell["color"], cell["kind"])
            self.board.move_history = []
            self.board.halfmove_clock = 0
        else:
            self.board = Board()
        self.board_history = [copy.deepcopy(self.board)]
        self.last_move_dest = None
        self.ai_game_moves = []
        if self.game_mode.get()=="HvA":
            self.human_color = "w"
            self.turn = "w"
            self.ai = ChessAI("b", max_depth=self.ai_depth, time_limit=self.ai_time_limit)
            self.ai_white = None
            self.ai_black = None
        elif self.game_mode.get()=="HvA_Black":
            self.human_color = "b"
            self.turn = "w"
            self.ai = ChessAI("w", max_depth=self.ai_depth, time_limit=self.ai_time_limit)
            self.ai_white = None
            self.ai_black = None
        elif self.game_mode.get()=="HvH":
            self.human_color = None
            self.turn = "w"
            self.ai = None
            self.ai_white = None
            self.ai_black = None
        elif self.game_mode.get()=="AvA":
            self.human_color = None
            self.turn = "w"
            self.ai_white = ChessAI("w", max_depth=self.ai_depth, time_limit=self.ai_time_limit)
            self.ai_black = ChessAI("b", max_depth=self.ai_depth, time_limit=self.ai_time_limit)
            self.ai = None
        elif self.game_mode.get()=="Custom":
            self.human_color = "w"
            self.turn = "w"
            self.ai = ChessAI("b", max_depth=self.ai_depth, time_limit=self.ai_time_limit)
            self.ai_white = None
            self.ai_black = None
        self.selected = None
        self.valid_moves = []
        self.draw_board()
        self.draw_pieces()
        self.update_status()
    def update_status(self):
        mode = self.game_mode.get()
        if mode in ["HvA","HvA_Black"]:
            if self.turn==self.human_color:
                self.status_label.config(text=self.your_turn_text + ("White)" if self.turn=='w' else "Black)"))
            else:
                self.status_label.config(text=self.ai_turn_text + ("White)" if self.turn=='w' else "Black)"))
        elif mode=="HvH":
            self.status_label.config(text=self.turn_text + ("White" if self.turn=='w' else "Black"))
        elif mode in ["AvA","Custom"]:
            self.status_label.config(text="AI (" + ("White" if self.turn=='w' else "Black") + ") is thinking...")
    def draw_board(self):
        self.canvas.delete("all")
        for r in range(8):
            for c in range(8):
                x1 = c*SQUARE_SIZE
                y1 = r*SQUARE_SIZE
                x2 = x1+SQUARE_SIZE
                y2 = y1+SQUARE_SIZE
                fill_color = self.light_color if (r+c)%2==0 else self.dark_color
                self.canvas.create_rectangle(x1,y1,x2,y2,fill=fill_color)
        if self.last_move_dest:
            r,c = self.last_move_dest
            x1 = c*SQUARE_SIZE
            y1 = r*SQUARE_SIZE
            x2 = x1+SQUARE_SIZE
            y2 = y1+SQUARE_SIZE
            self.canvas.create_rectangle(x1,y1,x2,y2,fill="#add8e6")
        if self.selected and self.show_moves:
            for (r,c) in self.valid_moves:
                x1 = c*SQUARE_SIZE + SQUARE_SIZE//4
                y1 = r*SQUARE_SIZE + SQUARE_SIZE//4
                x2 = x1+SQUARE_SIZE//2
                y2 = y1+SQUARE_SIZE//2
                if self.move_highlight_shape=="circle":
                    self.canvas.create_oval(x1,y1,x2,y2,outline=self.highlight_color,width=3)
                else:
                    self.canvas.create_rectangle(c*SQUARE_SIZE, r*SQUARE_SIZE, c*SQUARE_SIZE+SQUARE_SIZE, r*SQUARE_SIZE+SQUARE_SIZE, outline=self.highlight_color,width=3)
    def draw_pieces(self):
        for r in range(8):
            for c in range(8):
                piece = self.board.get_piece(r,c)
                if piece:
                    x = c*SQUARE_SIZE + SQUARE_SIZE//2
                    y = r*SQUARE_SIZE + SQUARE_SIZE//2
                    color = self.piece_color_white if piece.color=='w' else self.piece_color_black
                    self.canvas.create_text(x,y, text=PIECE_UNICODE[(piece.color,piece.kind.upper())], font=('Arial',32), fill=color)
    def on_click(self, event):
        mode = self.game_mode.get()
        if mode in ["HvA","HvA_Black"]:
            if self.turn!=self.human_color:
                return
        elif mode=="AvA" or mode=="Custom":
            return
        elif mode=="AvA" or mode=="Custom":
            return
        col = event.x//SQUARE_SIZE
        row = event.y//SQUARE_SIZE
        if self.selected:
            if (row,col) in self.valid_moves:
                self.board.move_piece(self.selected[0], self.selected[1], row, col, play_sound=False)
                self.last_move_dest = (row, col)
                self.board_history.append(copy.deepcopy(self.board))
                self.selected = None
                self.valid_moves = []
                self.toggle_turn()
                self.draw_board()
                self.draw_pieces()
                self.update_status()
                self.check_game_over()
                return
            else:
                self.selected = None
                self.valid_moves = []
        piece = self.board.get_piece(row, col)
        if piece and (mode=="HvH" or (mode in ["HvA","HvA_Black"] and piece.color==self.human_color)):
            self.selected = (row, col)
            self.valid_moves = self.board.get_valid_moves(row, col)
        self.draw_board()
        self.draw_pieces()
    def toggle_turn(self):
        self.turn = enemy(self.turn)
    def undo_move(self):
        if len(self.board_history)>1:
            self.board_history.pop()
            self.board = copy.deepcopy(self.board_history[-1])
            self.toggle_turn()
            self.last_move_dest = None
            self.selected = None
            self.valid_moves = []
            self.draw_board()
            self.draw_pieces()
            self.update_status()
    def game_loop(self):
        mode = self.game_mode.get()
        ai_player = None
        if mode=="AvA" or mode=="Custom":
            ai_player = self.ai_white if self.turn=='w' else self.ai_black
        elif mode in ["HvA","HvA_Black"] and self.turn!=self.human_color:
            ai_player = self.ai
        if ai_player is not None:
            move = ai_player.choose_move(self.board)
            if move:
                key = memory_key_to_str((get_board_key(self.board), move))
                self.ai_game_moves.append(key)
                sr, sc = move[0]
                dr, dc = move[1]
                self.board.move_piece(sr, sc, dr, dc, play_sound=False)
                self.last_move_dest = (dr, dc)
                self.board_history.append(copy.deepcopy(self.board))
                self.toggle_turn()
                self.draw_board()
                self.draw_pieces()
                self.update_status()
                self.check_game_over()
        self.root.after(500, self.game_loop)
    def update_ai_memory(self, reward):
        for key in self.ai_game_moves:
            ai_memory[key] = ai_memory.get(key,0) + reward
        save_ai_memory()
        self.ai_game_moves = []
    def check_game_over(self):
        legal_exists = False
        for r in range(8):
            for c in range(8):
                piece = self.board.get_piece(r,c)
                if piece and piece.color==self.turn:
                    if self.board.get_valid_moves(r,c):
                        legal_exists = True
                        break
            if legal_exists:
                break
        if not legal_exists:
            if self.board.is_in_check(self.turn):
                result = self.checkmate_text + (self.win_black_text if self.turn=='w' else self.win_white_text)
                if self.game_mode.get() in ["HvA","HvA_Black"] and self.ai is not None:
                    ai_color = self.ai.color
                    winner = enemy(self.turn)
                    if winner==ai_color:
                        self.update_ai_memory(30)
                    else:
                        self.update_ai_memory(-30)
            else:
                result = self.draw_text
            messagebox.showinfo(self.title_text, result)
            self.new_game()
    def train_ai(self):
        num_games = simpledialog.askinteger("Train AI", "Enter number of training games:", parent=self.root, minvalue=1, initialvalue=100)
        if num_games is None:
            return
        self.status_label.config(text="Training AI...")
        self.root.update()
        threading.Thread(target=self.train_ai_thread, args=(num_games,2,1), daemon=True).start()
    def train_ai_thread(self, num_games, depth, time_limit):
        simulate_ai_vs_ai_game(num_games=num_games, depth=depth, time_limit=time_limit)
        self.root.after(0, lambda: self.status_label.config(text="Training complete!"))
        self.root.after(0, lambda: messagebox.showinfo(self.title_text, f"Training complete: {num_games} games simulated."))

if __name__ == "__main__":
    root = tk.Tk()
    game = ChessTkinterGame(root)
    root.mainloop()