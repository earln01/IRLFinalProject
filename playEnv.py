import tensorflow as tf
import numpy as np
from tensorflow import keras
import chess
import os
import sys
import asyncio
import chess.engine
import chess.pgn
import chess.svg
import time
from trainEnv import ChessEnv

class playEnv(ChessEnv):
    def __init__(self, openingMoves, modelDir):
        self.model = keras.models.load_model(modelDir)
        self.board = chess.Board()
        self.openingMoves = openingMoves
        piece_types = ["P", "N", "B", "R", "Q", "K"]
        self.pieceVals = {piece_type: i+1 for i, piece_type in enumerate(piece_types)}
        self.reset()

    def reset(self):
        return super().reset()
    
    def getState(self):
        return super().getState()
    
    def playMove(self, move):
        super().playMove(move)
    
    def getResultingState(self, move):
        self.board.push(move)
        state = self.getState()
        self.board.pop()
        return state

    
    def getBestMove(self):
        bestMove = None
        bestScore = -sys.maxsize
        for move in self.board.legal_moves:
            nextState = self.getResultingState(move)
            tensor = tf.convert_to_tensor([nextState])
            score = np.amax(self.model.predict(tensor))
            if  score > bestScore:
                bestScore = score
                bestMove = move
        return bestMove
    
    def playBestMove(self):
        move = self.getBestMove()
        self.playMove(move)
        return move
    
    def getTurn(self):
        return self.board.turn

def main():
    openingMoves = ['e4', 'Nf6']
    modelDir = 'testModel'
    svgFile = 'result.svg'
    moveList = openingMoves
    player = playEnv(openingMoves, modelDir)
    while not player.board.is_game_over():
        move = player.playBestMove()
        moveList.append(move.uci())
        print(moveList)
    
    boardSvg = chess.svg.board(player.board, size=350)
    with open(svgFile, 'w') as outfile:
        outfile.write(boardSvg)
    time.sleep(0.1)
    os.startfile(svgFile)

if __name__ == '__main__':
    main()