from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import SDNEnvironmentOnos as env
import time
import os
import tensorflow as tf
from tensorflow import keras
import progressbar
import numpy as np
import random
from collections import deque, namedtuple
from tensorflow._api.v1.compat.v2.summary import experimental
actions=list(range(0,10))
batch_size = 32
epsilon = 1.0

class Agent:
  global epsilon
  def __init__(self,is_training,double_q,Dueling):

    # Initialize atributes
    self.expirience_replay = deque(maxlen=10000)
    self._is_training = is_training
    self.double_q=double_q
    self.Dueling=Dueling
    # Initialize discount and exploration rate
    self.gamma = 0.6


    # Build networks
    if os.path.isfile("weights/q_network.h5"):   # load model if its already there
      self.q_network = keras.models.load_model('weights/q_network.h5')
      self.target_network = keras.models.load_model('weights/target_network.h5')
      print('models loaded')
    else:
      self.q_network = self._build_compile_model()
      self.target_network = self._build_compile_model()

    #self.alighn_target_model()


  def store(self, state, action, reward, next_state):
    self.expirience_replay.append((state, action, reward, next_state))


  def _build_compile_model(self):
    if self.Dueling:
       flatten_value=  keras.Input(shape=(1,))
       value_func_hidden   = keras.layers.Dense(128, activation=tf.nn.relu)(flatten_value)
       value_func= keras.layers.Dense(1, activation=tf.nn.softmax)(value_func_hidden)
       advantage_func_hidden_1 = keras.layers.Dense(128, activation=tf.nn.relu)(flatten_value)
       advantage_func_hidden_2 = keras.layers.Dense(1, activation=tf.nn.softmax)(advantage_func_hidden_1)
       advantage_func = keras.layers.Dense(10, activation=tf.nn.softmax)(advantage_func_hidden_2)


       # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
       output = value_func + (advantage_func - tf.reduce_mean(advantage_func, axis=1, keepdims=True))
       model = keras.Model(inputs=flatten_value, outputs=output)
       model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.5),
                     loss='mse',
                     metrics=['accuracy'])
       return model
    else:
      model = keras.Sequential([
        keras.layers.Flatten(input_shape=(1,)),  # (7,)
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu), # atleast 2 dense layers create a deep neural network
        keras.layers.Dense(10, activation=tf.nn.softmax)   # 10 classes we want to predict
      ])

      model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.5),
                  loss='mse',
                  metrics=['accuracy'])
      return model

  def alighn_target_model(self):
    self.target_network.set_weights(self.q_network.get_weights())

  def save_models(self):
      self.q_network.save('weights/q_network.h5')
      self.target_network.save('weights/target_network.h5')

  def act(self, state):
    global epsilon
    if self._is_training:
      if np.random.rand() <= epsilon:
        return random.choice(actions)
      else:
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])
    else:   # while not evaluation just take the best value and no exploration
      q_values = self.q_network.predict(state)
      return np.argmax(q_values[0])

  def get_metrics(self,step):
    self._step= step
    loss=self.metrics.get("loss", None)
    accuracy = self.metrics.get("accuracy", None)
    with open('weights/loss.txt', 'a+') as files:
      files.write(str(loss) + "\n")
      files.close()
    with open('weights/accuracy.txt', 'a+') as files:
      files.write(str(accuracy) +  "\n")
      files.close()
    return self.metrics

  #@tf.function
  def retrain(self, batch_size):
    self._batchsize=batch_size
    minibatch = random.sample(self.expirience_replay, self._batchsize)
    if self.double_q:
      #print('double')
      for state, action, reward, next_state in minibatch:
        # predict Q-value of current state
        target_value = self.q_network.predict(state)
        #print(self.q_network.predict(next_state))
        #print(self.q_network.predict(next_state)[0]) =[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
        predict_action_value = np.argmax(self.q_network.predict(next_state)[0])
        target_value[0][action] = reward + self.gamma * self.target_network.predict(next_state)[0][predict_action_value]
        History = self.q_network.fit(state, target_value, epochs=1, verbose=0)  # ,callbacks=[tensorboard_callback]
      self.metrics = History.history
    else:  #DQN
      for state, action, reward, next_state in minibatch:
        #predict Q-value of current state
        predict = self.q_network.predict(state)
        #predict Q-value of Next state
        target = self.target_network.predict(next_state)
        #biased target value calculation
        predict[0][action] = reward + self.gamma * np.amax(target)
        #train Q_Network
        History = self.q_network.fit(state, predict, epochs=1, verbose=0)#,callbacks=[tensorboard_callback]
      self.metrics= History.history



def q_learning(state,total):
  total_reward = total
  current_state = state
  action = agent.act(current_state)
  with open('weights/action_list.txt', 'a+') as files:
    files.write(str(action) + "\n")
    files.close()
  time_step = environment.step(action)
  next_state = time_step.observation
  reward = time_step.reward
  with open('weights/immediate_reward.txt', 'a+') as files:
    files.write(str(reward) + "\n")
    files.close()
  total_reward = total_reward + reward
  agent.store(current_state, action, reward, next_state)
  return  next_state,total_reward


print(tf.__version__)
print(tf.executing_eagerly())
if __name__ == '__main__':
  total_reward = 0
  environment = env.SDNEnvironment()
  num_of_episodes = 100
  is_evaluating = False
  reward_frequency = 0
  agent = Agent(True,True,False)  # agent is training when we will be testing it will be False
  episode = 0
  time_step = environment.reset()
  current_state = time_step.observation
  while(len(agent.expirience_replay)< 35):   # precollecting experiences with randomly taking action
    action= random.choice(actions)
    time_step = environment.step(action)
    next_state = time_step.observation
    reward = time_step.reward
    agent.store(current_state, action, reward, next_state)
    current_state = next_state
  while(episode<=num_of_episodes):
    # Reset the environment
    time_step = environment.reset()
    current_state = time_step.observation
    if episode == 0:
      agent.alighn_target_model()
##########
    episode_running = True
    while(episode_running):
      if is_evaluating:
        if (current_state[2] == -1) or (current_state[3] == -1) :    ## it means things are bad already no need to check threshold
            next_state,total_reward = q_learning(current_state,total_reward)
            reward_frequency = reward_frequency + 1
            current_state = next_state
      else:
        next_state,total_reward = q_learning(current_state,total_reward)
        reward_frequency = reward_frequency + 1
        current_state = next_state
      if environment.episode_ended:
        episode_running = False
        environment.episode_ended=False
        agent.alighn_target_model()
        episode = episode + 1
      if len(agent.expirience_replay) > batch_size:
        agent.retrain(batch_size)
    if episode == num_of_episodes:
      agent.save_models()
      print("Model Saved")
      print("**********************************")
    if (episode ) % 1 == 0:   # save model every 5 epsiodes
      print("**********************************")
      print('size', len(agent.expirience_replay))
      print("Episode: {}".format(episode))
      epsilon = epsilon - 0.01
      print('history dict:', agent.get_metrics(episode))
      with open('weights/avg_reward.txt', 'a+') as files:
        files.write(str(total_reward/reward_frequency) + "\n")
        files.close()
      total_reward = 0

