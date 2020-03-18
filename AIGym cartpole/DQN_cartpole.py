import gym
from collections import namedtuple
import tensorflow as tf
from tensorflow import keras
import numpy as np

from collections import deque
from random import seed, sample, randint


BATCH_SIZE = 30
GAMMA = 0.999
TARGET_UPDATE = 10
MAX_EXPERIENCE = 1000
MIN_EXPERIENCE = 200
LEARNING_RATE = 0.00025


'''
Acts as circular buffer, which stores each state
'''
class Memory:

    def isFull(self):
        return self.index + 1 == self.__size

    def getCurrentSize(self):
        return self.index

    def getState(self):
        state = self.buffer.pop()
        self.buffer.append(state)
        return state

    def __init__(self, memory_size):
        seed()
        self.__size = memory_size
        self.index = 0
        self.buffer = deque(maxlen=memory_size)

    def addState(self, action, observation, reward, next_state, notDone):
        current_state = State(
            action=action,
            reward=reward,
            observation=observation,
            next=next_state,
            notDone=notDone)

        self.buffer.append(current_state)
        self.index += 1

    def removeBatch(self, batch_size):
        for i in range(batch_size):
            self.buffer.popleft()
            self.index -= 1


    def getRandomBatch(self, batch_size):
        random_states = sample(self.buffer, batch_size)

        start_states = []
        next_states = []
        rewards = []
        actions = []
        notDone = []
        for x in random_states:
            start_states.append([x.position, x.velocity, x.angle, x.rotation_rate])
            next_states.append([x.next])
            rewards.append(x.reward)
            actions.append(x.action)
            notDone.append(x.notDone)


        return np.array(start_states), np.squeeze(np.array(next_states)), np.array(rewards), np.squeeze(np.array(actions)), np.array(notDone)




class State:

    def __init__(self, action, reward, observation, next, notDone):
        self.reward = reward
        self.action = action
        self.position, self.velocity, self.angle, self.rotation_rate = observation[0]
        self.next = next
        self.notDone = notDone


def fit_batch(model, gamma, start_states, actions, rewards, next_states, done):
    """

    @params
    model:          DQN model
    gamma:          0.99
    start_states:   observation
    actions:        actions (0,1)
    rewards:        rewards (0,1)
    next_states:    resulting states (observation)

    """
    # print("next states: ")
    # print(next_states)
    next_Q_values = model.predict([next_states])
    # print("next_Q_values: ")
    # print(next_Q_values)

    Q_values = rewards + gamma * np.max(next_Q_values, axis=1) * done
    Q_values = actions * Q_values[:, None]
    # print("Q_values: ")
    # print(Q_values)

    model.fit(
        start_states, Q_values,
        epochs=1, verbose=0
    )


def choose_best_action(model, state):
    return model.predict(state)


def q_iteration(env, model, state, iteration, memory, episode_num):
    global epsilon
    prev_state = state
    action = choose_best_action(model, state)
    # print(action)

    epsilon = max(min_epsilon, epsilon * decay_epsilon)

    if (random() < epsilon):
        action = [[random(), random()]]
    env.render()

    observation, reward, done, info = env.step(np.argmax(action))

    if done:
        if not memory.isFull():
            memory.addState(action, prev_state, reward, observation, 0)
            print("Episode finished after {} timesteps".format(t+1))
        return True, state

    if memory.isFull():
        print("memory is full")
        start_states, next_states, rewards, actions, notDone = memory.getRandomBatch(BATCH_SIZE)
        memory.removeBatch(BATCH_SIZE)
        fit_batch(model,
                gamma=0.99,
                start_states=start_states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                done=notDone)

    elif memory.getCurrentSize() > MIN_EXPERIENCE:
        memory.addState(action, prev_state, reward, observation, 1)
        start_states, next_states, rewards, actions, notDone = memory.getRandomBatch(BATCH_SIZE)
        fit_batch(model,
                gamma=0.99,
                start_states=start_states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                done=notDone)
    else:
        memory.addState(action, prev_state, reward, observation, 1)

    return False, observation.reshape(1,4)






HistoryTuple = namedtuple('action', ('position', 'velocity', 'angle', 'rotation_rate', 'result'))

env = gym.make('CartPole-v0')

from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

from random import seed, random

inputs = keras.Input(shape=(4,), name='observations')
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
output = layers.Dense(2, activation='linear', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=output)
optimizer = tf.optimizers.RMSprop(lr=LEARNING_RATE)
loss = tf.losses.mean_squared_error
model.compile(optimizer=optimizer,
        loss=loss)

memory = Memory(MAX_EXPERIENCE)

prev_observation = env.reset()

epsilon = 0.9999
min_epsilon = 0.05
decay_epsilon = 0.99999

consecutive_successes = 0
for i_episode in range(100):
    print("-----")
    print("Episode " + str(i_episode))
    print("-----")

    observation = env.reset()
    done = False

    state = observation.reshape(1,4)
    for t in range(450):
        done, state = q_iteration(env=env, model=model, state=state, iteration=t, memory=memory, episode_num=i_episode)
        if done:
            break
        if t > 195:
            consecutive_successes += 1
            print("195 reward reached! Yee!")
            break
        else:
            consecutive_successes = 0

env.close()