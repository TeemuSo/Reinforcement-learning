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

    def __init__(self, memory_size):
        seed()
        self.index = 0
        self.buffer = deque(maxlen=memory_size)

    def addState(self, action, observation, reward, next_state):
        current_state = State(
            action=action,
            reward=reward,
            observation=observation,
            next=next_state)

        self.buffer.append(current_state)
        self.index += 1

    def getRandomBatch(self, batch_size):
        random_states = sample(self.buffer, batch_size)
        for i in range(batch_size):
            self.buffer.popleft()



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

memory = Memory(30)

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

