import os
import asyncio
import chess
import chess.engine
import chess.pgn
import sqlite3
import random

SELF_DIR = os.path.dirname(os.path.realpath(__file__))
ENGINE_PATH = os.path.join(SELF_DIR, 'stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe')

conn = sqlite3.connect('chess.db')




async def main() -> None:
    board = chess.Board()
    transport, engine = await chess.engine.popen_uci(ENGINE_PATH)
    while not board.is_game_over():
        candidates = []
        if board.turn:
            print('White to move')
        else:
            print('Black to move')
        topMoves = await engine.analyse(board, chess.engine.Limit(time=0.1) , multipv=3)
        for move in topMoves:
            candidates.append(move['pv'][0])
        nextMove = candidates[random.randint(0,2)]
        print(nextMove)
        board.push(nextMove)
    await engine.quit()


def PGNtoDB(filePath):
    c = conn.cursor()
    createCMD = 'CREATE TABLE IF NOT EXISTS games ([result] TEXT, [moves] TEXT)'
    c.execute(createCMD)
    conn.commit()

    with open(filePath) as pgnFile:
        insertCMD = "INSERT into games (result, moves) VALUES (?, ?)"
        
        while True:
            pgn = chess.pgn.read_game(pgnFile)
            if pgn is not None:
                c.execute(insertCMD, (pgn.headers['Result'], str(pgn.mainline_moves())))
            else:
                break

asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
asyncio.run(main())
