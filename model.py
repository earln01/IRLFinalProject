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
    def __init__(self):
        return None

    async def init(self) -> None:
        self.env = ChessEnv()
        await self.env.init()

        ## TODO make this some sort of actual useful NN
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(units=64, input_shape=(8,8,), activation='linear'))

        self.model.compile(optimizer='adam', loss='mse')
    
    async def train(self, numEpisodes=1000, gamma=0.1, epsilon=0.1):
        for episode in range(numEpisodes):
            state = self.env.reset()
            done=False
            print(episode)
            while not done:
                randMove, randState = self.env.getRandomMoveandState()
                bestMove, bestState = await self.env.getBestMoveandState()

                if random.random() >= epsilon:
                    nextMove = bestMove
                    nextState = bestState
                else:
                    nextMove = randMove
                    nextState = randState

                reward = 1 if nextMove==bestMove else -1

                # Calculate the target Q-value for the current state
                target = reward + gamma * np.amax(self.model.predict(tf.convert_to_tensor([nextState])))

                # Update the model's Q-values for the current state
                self.model.fit(tf.convert_to_tensor([state]), tf.expand_dims(tf.convert_to_tensor([target]),  axis=1), batch_size=1,verbose=0)

                # Set the current state to the next state
                self.env.playMove(nextMove)
                state = nextState

                done = self.env.board.is_game_over()
        await self.env.tearDown()



async def main() -> None:
    learner = DQN()
    await learner.init()
    await learner.train()

if __name__ == '__main__':
    asyncio.run(main())