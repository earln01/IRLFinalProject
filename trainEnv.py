import os
import sys
import asyncio
import chess
import chess.engine
import chess.pgn
import random
import numpy as np
import sqlite3

SELF_DIR = os.path.dirname(os.path.realpath(__file__))
ENGINE_PATH = os.path.join(SELF_DIR, 'stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe')

class ChessDB:
    def __init__(self, dbName='chess.db'):
        self.dbName = dbName
        self.conn = sqlite3.connect(self.dbName)

    def PGNtoDB(self, pgnFile, drop=False):
        cursor = self.conn.cursor()
        if drop:
            cursor.execute('DROP TABLE IF EXISTS games')
        cursor.execute('CREATE TABLE IF NOT EXISTS games ([result] TEXT, [positions] TEXT, [length] INT)')
        insertCMD = "INSERT into games (result, positions, length) VALUES (?, ?, ?)"

        with open(pgnFile) as f:
            
            while True:
                pgn = chess.pgn.read_game(f)
                if pgn is not None:
                    positions = self.positionsFromMoves(pgn.mainline_moves())
                    cursor.execute(insertCMD, (pgn.headers['Result'], str(positions), len(list(pgn.mainline_moves()))))
                else:
                    break
        self.conn.commit()
    
    def positionsFromMoves(self, moves):
        board = chess.Board()
        positions=[]
        for move in moves:
            board.push(move)
            positions.append(board.board_fen())
        return positions
    
    def getMoveScore(self, pre_position, post_position, white=True):
        winString = '1-0' if white else '0-1'
        loseString = '0-1' if white else '1-0'
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM games WHERE positions LIKE ?", ("%" + pre_position + "', '" +  post_position + "%",))
        score = 0
        games = 0
        
        for row in cursor:
            if row[0] == winString:
                score += (100-row[2])
            elif row[0] == loseString:
                score -= (100 + row[2])
            else:
                score -= 100
            games += 1
        score = score/games if games >= 5 else -200
        return score
    
    def getNumGames(self, position):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM games WHERE positions LIKE ?", ("%" + position + "%",))
        return len(list(cursor)) 

    def tearDown(self):
        self.conn.close()       

class ChessEnv:
    def __init__(self, pgnFiles, openingMoves):
        self.board = chess.Board()
        self.openingMoves = openingMoves
        piece_types = ["P", "N", "B", "R", "Q", "K"]
        self.pieceVals = {piece_type: i+1 for i, piece_type in enumerate(piece_types)}
        self.reset()
        self.chess_db = ChessDB()
        for file in pgnFiles:
            self.chess_db.PGNtoDB(file)

    async def init(self) -> None:
        self.transport, self.engine = await chess.engine.popen_uci(ENGINE_PATH)
    
    def reset(self):
        self.board.reset()
        self.board.clear_stack()
        for move in self.openingMoves:
            self.board.push_san(move)
        return self.getState()
    
    def getState(self):
        arr = np.zeros((8, 8), dtype=np.int8)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                arr[square // 8][square % 8] = self.pieceVals[piece.symbol().upper()]
        return arr
    
    async def getBestMoveandState(self):
        if self.chess_db.getNumGames(self.board.board_fen()) < 5:
            bestMove = await self.engine.play(self.board, chess.engine.Limit(time=0.1))
            bestMove = bestMove.move
            self.board.push(bestMove)
            state = self.getState()
            self.board.pop()
            return (bestMove, state)
        else:
            white = self.board.turn
            pre_position = self.board.board_fen()
            topMoves = await self.engine.analyse(self.board, chess.engine.Limit(time=0.1) , multipv=3)
            candidates = [move['pv'][0] for move in topMoves]
            bestScore = -sys.maxsize
            bestMove = None
            for candidate in candidates:
                self.board.push(candidate)
                score = self.chess_db.getMoveScore(pre_position, self.board.board_fen(), white)
                if score > bestScore:
                    bestScore = score
                    bestMove = candidate
                self.board.pop()
            self.board.push(bestMove)
            state = self.getState()
            self.board.pop()
            return(bestMove, state)

    def getRandomMoveandState(self):
        randMove = random.choice(list(self.board.legal_moves))
        self.board.push(randMove)
        state = self.getState()
        self.board.pop()
        return(randMove, state)
    
    def playMove(self, move):
        self.board.push(move)
    
    async def tearDown(self):
        await self.engine.quit()
        self.chess_db.tearDown()
