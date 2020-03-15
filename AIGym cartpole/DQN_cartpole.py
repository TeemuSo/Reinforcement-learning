import gym
from collections import namedtuple
import tensorflow as tf
from tensorflow import keras
import numpy as np

from collections import deque
from random import seed, sample

'''
Acts as circular buffer, which stores each state
'''
class Memory:

    def isFull(self):
        return self.index - 1 == self.__size

    def getState(self, index):
        return self.buffer

    def __getitem__(self, idx):
        return self.buffer[(self.start + idx) % len(self.buffer)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.buffer) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __init__(self, memory_size):
        seed()
        self.start = 0
        self.end = 0
        self.buffer = [None] * (memory_size + 1)


    def addState(self, action, observation, reward, next_state):
        current_state = State(
            action=action,
            reward=reward,
            observation=observation,
            next=next_state)

        self.buffer[self.end] = current_state
        self.end = (self.end + 1) % len(self.buffer)

    def getRandomBatch(self, batch_size):
        random_states = sample(self.buffer, batch_size)



    def storeSequence(self, ):
        a,b,c,d = observation.tolist()
        self.memory.append(HistoryTuple(a,b,c,d,result))


class State:

    def __init__(self, action, reward, observation, next):
        self.reward = reward
        self.action = action
        self.position, self.velocity, self.angle, self.rotation_rate = observation
        self.next = next




HistoryTuple = namedtuple('action', ('position', 'velocity', 'angle', 'rotation_rate', 'result'))

env = gym.make('CartPole-v0')
observation = env.reset()

from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from random import seed, random

inputs = keras.Input(shape=(4,), name='observations')
x = layers.Dense(10, activation='relu', name='dense_1')(inputs)
x = layers.Dense(10, activation='relu', name='dense_2')(x)
output = layers.Dense(2, activation='sigmoid', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam',
        loss=keras.losses.binary_crossentropy)

memory = Memory(50)
seed(1)
for i_episode in range(100):
    observation = env.reset()
    done = False


    for t in range(450):
        env.render()
        action = np.argmax(model.predict(observation.reshape(1,4)))

        if random() > i_episode*0.02:
            action = 1-action

        observation, reward, done, info = env.step(action)

        if done:
            memory.storeSequence(action, observation, 1-action)
            training_data = np.array(memory.memory)[:,:4]
            training_labels = np.array(memory.memory)[:,4]
            training_labels = to_categorical(training_labels)
            model.fit(training_data, training_labels, epochs=1)

            print("Episode finished after {} timesteps".format(t+1))
            break
        else:
            if observation[2] >
            memory.storeSequence(action, observation, action)
env.close()

