import os
import asyncio
import chess
import chess.engine
import chess.pgn
import random
import numpy as np

SELF_DIR = os.path.dirname(os.path.realpath(__file__))
ENGINE_PATH = os.path.join(SELF_DIR, 'stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe')

class ChessEnv:
    def __init__(self):
        return None

    async def init(self, action_space) -> None:
        self.transport, self.engine = await chess.engine.popen_uci(ENGINE_PATH)
        self.board = chess.Board()
        self.action_space = action_space
        piece_types = ["P", "N", "B", "R", "Q", "K"]
        self.pieceVals = {piece_type: i+1 for i, piece_type in enumerate(piece_types)}
    
    def reset(self):
        ## TODO start from a set opening
        self.board = chess.Board()
        return self.getState()
    
    def getState(self):
        arr = np.zeros((8, 8), dtype=np.int8)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                arr[square // 8][square % 8] = self.pieceVals[piece.symbol().upper()]
        return arr
    
    async def getBestMoveandState(self):
        ## TODO change this to reflect our best move method by searching database

        bestMove = await self.engine.play(self.board, chess.engine.Limit(time=0.1))
        bestMove = bestMove.move
        self.board.push(bestMove)
        state = self.getState()
        self.board.pop()
        return (bestMove, state)

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
