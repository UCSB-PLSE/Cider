# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
import numpy.ma as ma
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.01
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, mask):
        if np.random.rand() <= self.epsilon:
            legal = [i for i,f in enumerate(mask) if f]
            best = random.choice(legal)
            # print("Explore {}".format(best))
        else:
            act_values = self.model.predict(state)
            act_values = act_values[0]
            masked = ma.array(act_values, mask=~mask)
            best = ma.argmax(masked)
        return best

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in tqdm(minibatch):
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def solid():
    from environment import Environment
    # from environment import Environment
    filename = "../SolidTypes/test/regression/good/mint_MI.sol"
    # filename = "../SolidTypes/test/regression/good/standard_erc20.sol"
    env = Environment(filename)
    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)

    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # env.render()
            action = agent.act(state, mask=env.get_mask())
            next_state, reward, done, _ = env.step(action)
            # reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, reward: {}, e: {:.2}"
                    .format(e, EPISODES, reward, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")

if __name__ == "__main__":
    solid()
    
