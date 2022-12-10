import os
import asyncio
import chess
import chess.engine
import chess.pgn
import random
import tensorflow as tf
import numpy as np
import random
from trainEnv import ChessEnv

asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())

class DQN:
    def __init__(self, pgnFiles, openingMoves):
        self.pgnFiles = pgnFiles
        self.openingMoves = openingMoves

    async def init(self) -> None:
        self.env = ChessEnv(pgnFiles = self.pgnFiles, openingMoves=self.openingMoves)
        await self.env.init()

        ## TODO make this some sort of actual useful NN
        self.whiteModel = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', input_shape=(8,8,1)),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='linear')
            ])

        self.blackModel = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', input_shape=(8,8,1)),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='linear')
            ])

        self.whiteModel.compile(optimizer='adam', loss='mse')
        self.blackModel.compile(optimizer='adam', loss='mse')
    
    async def train(self, numEpisodes=1000, gamma=0.1, epsilon=0.1):
        for episode in range(numEpisodes):
            done=False
            self.env.reset()
            print(episode)
            while not done:

                if random.random() >= epsilon:
                    nextMove, nextState, eval = await self.env.getBestMoveandState()
                    
                else:
                    nextMove, nextState, eval = await self.env.getRandomMoveandState()
                    

                # Calculate the target Q-value for the current state
                # target = curReward + gamma * np.amax(self.model.predict(tf.convert_to_tensor([nextState])))

                # Update the model's Q-values for the current state

                ## Supervised learning - map states to reward
                # To switch back to Q learning - self.model.fit(tf.convert_to_tensor([state]), tf.expand_dims(tf.convert_to_tensor([target]),  axis=1), batch_size=1,verbose=0)
                if self.env.board.turn:
                    self.whiteModel.fit(tf.convert_to_tensor([nextState]), tf.expand_dims(tf.convert_to_tensor([eval]),  axis=1), batch_size=1,verbose=0)
                else:
                    self.blackModel.fit(tf.convert_to_tensor([nextState]), tf.expand_dims(tf.convert_to_tensor([eval]),  axis=1), batch_size=1,verbose=0)

                # Set the current state to the next state
                self.env.playMove(nextMove)

                done = self.env.board.is_game_over()
        await self.env.tearDown()
    
    def saveModel(self, save_directory_white, save_directory_black):
        self.whiteModel.save(save_directory_white)
        self.blackModel.save(save_directory_black)


async def main() -> None:
    pgnFiles = ['alekhine.pgn']
    openingMoves = ['e4', 'Nf6']
    save_directory_white = 'testModelWhite'
    save_directory_black = 'testModelBlack'
    learner = DQN(pgnFiles, openingMoves)
    await learner.init()
    await learner.train()
    learner.saveModel(save_directory_white, save_directory_black)

if __name__ == '__main__':
    asyncio.run(main())