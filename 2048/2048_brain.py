from tkinter import *
from random import *
import tensorflow as tf      # Deep Learning library
import numpy as np
import time
import matplotlib
from collections import deque
import os



matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, './2048-python')
from puzzle import *
from logic import *

#Hyper parameters
state_size = [4,4]
action_size = [4]
learning_rate =  0.0002
max_steps = 10000

up = [1,0,0,0]
down = [0,1,0,0]
left = [0,0,1,0]
right = [0,0,0,1]
possible_actions = ["'w'", "'s'", "'a'", "'d'"]
actions_choice = [0,1,2,3]

total_episodes = 5000      # Total episodes for training
# max_steps = 100              # Max possible steps in an episode
batch_size = 32

explore_start = 0.90        # exploration probability at start
explore_stop = 0.0001            # minimum exploration probability
decay_rate = 0.0001

gamma = 0.8               # Discounting rate

pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)

        return [self.buffer[i] for i in index]

class DQNetwork:
  def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            # print(matrix_state)
            self.inputs_ = tf.placeholder(tf.float32, [None, 16], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 4], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            self.fc = tf.layers.dense(inputs = self.inputs_,
                                  units = 16,
                                  activation = tf.nn.elu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name="fc1")

            self.fc = tf.layers.dense(inputs = self.fc,
                                  units = 16,
                                  activation = tf.nn.elu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name="fc2")
            self.fc = tf.layers.dense(inputs = self.fc,
                                  units = 16,
                                  activation = tf.nn.elu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name="fc3")
            self.fc = tf.layers.dense(inputs = self.fc,
                                  units = 16,
                                  activation = tf.nn.elu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name="fc4")

            self.output = tf.layers.dense(inputs = self.fc,
                                  units = 4,
                                  activation = None,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name="fc5")

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            print("self.q is : ", self.Q)


            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            # print("self fc is : ", self.fc[1])
            # sess = tf.Session()

            # print(sess.run(self.fc))

# Instantiate memory
memory = Memory(max_size = memory_size)

# Render the environment


for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        # First we need a state
        state = tk.matrix

    # Random action
    useful_moves_list = useful_moves(state)
    index_action = np.random.choice(useful_moves_list)
    action = possible_actions[index_action]

    # Get the rewards
    reward = tk.ai_key_down(action)

    # Look if the episode is finished
    done = tk.game_finished()
    # print("been here")
    # If we're dead
    if done:
        # We finished the episode
        next_state = tk.init_matrix()
        # print("new game")

        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Start a new episode
        # game.new_episode()

        # First we need a state
        state = tk.matrix

        # Stack the frames
        # state, stacked_frames = stack_frames(stacked_frames, state, True)

    else:
        # Get the next state
        next_state = tk.matrix
        # next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Our state is now the next_state
        state = next_state

def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
  exp_exp_tradeoff = np.random.rand()
  random_flag = False
  explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
  # print("exploration prob : ", explore_probability)

  if(explore_probability > exp_exp_tradeoff):
    useful_moves_list = useful_moves(state)
    index_action = np.random.choice(useful_moves_list)
    action = possible_actions[index_action]
    random_flag = True
    # print("random is : ", action)


  else:
    Qs = sess.run(dq.output, feed_dict = {dq.inputs_: np.reshape(state, (1, 16))})
    useful_moves_list = useful_moves(state)
    possible_moves = []

    futur_correlation_list = tk.futur_correlation()

    mn_corr = array_mean_normalization(futur_correlation_list)
    mn_Qs = array_mean_normalization(Qs[0])

    ultra_instinct = np.add(mn_Qs,mn_corr)
    # print("ultra_instinct is : ", np.divide(mn_corr,2), " and ", mn_corr)

    for action in actions_choice:
      if action in useful_moves_list:
        possible_moves.append(ultra_instinct[action])
      else :
        possible_moves.append(-10000)
    # print("then ultra_instinct is : ", ultra_instinct)

    choice = np.argmax(possible_moves)
    # choice = np.argmax(ultra_instinct)
    # print("possible_moves are : ",possible_moves)

    # print("choice is : ", choice)
    # print("int-choice is : ", int(choice))
    action = possible_actions[int(choice)]
    #
    # print("action is : ", action)

  return action, explore_probability, random_flag


i = 0
# print("matrix state is : ", tk.matrix)
dq = DQNetwork(state_size, action_size, learning_rate)


with tf.Session() as sess:
    tk.update_idletasks()
    tk.update()
    # Initialize the variables
    sess.run(tf.global_variables_initializer())

    # Initialize the decay rate (that will use to reduce epsilon)
    decay_step = 0
    total_reward = []
    total_mean_reward = []
    list_20_ep_mean = [0]*20
    highest_tile = 0
    highest_record = [0,0,0,0]

    for episode in range(total_episodes):
        tk.init_matrix()
        tk.update_idletasks()
        tk.update()
        # time.sleep(0.1)

        # Set step to 0
        step = 0
        random_step = 0

        # Initialize the rewards of the episode
        episode_rewards = []

        # Make a new episode and observe the first state
        state = np.array(tk.matrix)

        # print(state)
        print("dq.loss is : ", dq.loss)
        print("episode is : ", episode)
        if(episode == 200):
          decay_rate = 0.0001
          explore_stop = 0.00001            # minimum exploration probability



        while step < max_steps:
            tk.update_idletasks()
            tk.update()
            step += 1

            # Increase decay_step
            decay_step +=1

            # Predict the action to take and take it
            action, explore_probability, random_flag = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)
            if random_flag: random_step+= 1
            # Do the action
            # reward = tk.ai_key_down(action)
            reward = tk.ai_key_down(action)
            # print("game score is : ", reward)

            # Look if the episode is finished
            done = tk.game_finished()
            # print("episode is : ", done)

            # Add the reward to total reward
            episode_rewards.append(reward)

            # If the game is finished
            if done:
                # the episode ends so no next state
                # next_state = np.zeros((4,4), dtype=np.int)
                # next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                # Set step = max_steps to end the episode
                print("steps is : ", step)
                print("random steps is : ", random_step)
                print(np.array(tk.matrix))

                flat = [item for sublist in state for item in sublist]
                # print("there ya go : ",flat)

                current_highest_tile = flat[np.argmax(flat)]

                if(current_highest_tile == 1024):
                  os.system('say "So close."')


                if(current_highest_tile == 2048):
                  os.system('say "You did it ! You fucking did it maaaaaan"')
                  os.system('say "Are you proud of yourself?"')
                  os.system('say "All this time just for that?"')





                # print("there ya : ",value)
                print("recap \n ")

                print("current highest tile is : \n", current_highest_tile, " ", np.sum(episode_rewards))
                print("recorded highest tile is : \n", highest_tile)
                if(highest_tile == current_highest_tile):
                  if(highest_record[1] < np.sum(episode_rewards)):
                    highest_tile = current_highest_tile
                    highest_record = []
                    highest_record = [highest_tile, np.sum(episode_rewards), step, random_step, episode]
                if(highest_tile < current_highest_tile):
                  highest_tile = current_highest_tile
                  highest_record = []
                  highest_record = [highest_tile, np.sum(episode_rewards), step, random_step, episode]

                step = max_steps

                # Get the total reward of the episode
                total_reward.append(np.sum(episode_rewards))
                total_mean_reward.append(np.mean(total_reward))
                list_20_ep_mean.append(np.mean(total_reward[-20:]))
                # print("total reward of episode is : ", total_reward)

                print('Episode: {}\n'.format(episode),
                          # 'Total reward: {}\n'.format(total_reward),
                          'Highest record: {}\n'.format(highest_record),
                          'episode score: {}\n'.format(np.sum(episode_rewards)),
                          'mean score since beginning: {:.4f}\n'.format(np.mean(total_reward)),
                          'mean score last 20 episodes: {:.4f}\n'.format(np.mean(total_reward[-20:])),
                          'Explore P: {:.4f}'.format(explore_probability))
                #
                memory.add((state, action, reward, next_state, done))

                # if(episode%5 == 0):
                #   plt.plot(total_reward)
                #   plt.plot(total_mean_reward)
                #   plt.plot(list_20_ep_mean)
                #   plt.ylabel('episode\'s reward')
                #   plt.show()

                print("\n------------------------------------------------------ \n")

            else:
                # Get the next state
                next_state = tk.matrix
                # print(next_state)

                # Stack the frame of the next_state
                # next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)


                # Add experience to memory
                memory.add((state, action, reward, next_state, done))

                # st+1 is now our current state
                state = next_state


            ### LEARNING PART
            # Obtain random mini-batch from memory
            batch = memory.sample(batch_size)
            # print("batch is : ", batch)
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            actions_mb = np.array([each[1] for each in batch])
            # print("actions_mb is : ", actions_mb)
            rewards_mb = np.array([each[2] for each in batch])
            next_states_mb = np.array([each[3] for each in batch], ndmin=3)
            # print("next_states_mb is : ", next_states_mb)

            dones_mb = np.array([each[4] for each in batch])

            target_Qs_batch = []


            # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
            for i in range(0, len(batch)):
                terminal = dones_mb[i]

                # If we are in a terminal state, only equals reward
                if terminal:
                    target_Qs_batch.append(rewards_mb[i])

                else:
                    state_mb = next_states_mb[i]

                    # Get Q values for next_state
                    Qs_next_state = sess.run(dq.output, feed_dict = {dq.inputs_: np.reshape(state_mb, (1,16))})
                    # print("rewards_mb ", rewards_mb)
                    # print("Qs_next_state ", Qs_next_state)
                    target = rewards_mb[i] + gamma * np.max(Qs_next_state[0])
                    target_Qs_batch.append(target)


            targets_mb = np.array([each for each in target_Qs_batch])

            # loss, _ = sess.run([dq.loss, dq.optimizer],
            #                     feed_dict={dq.inputs_: states_mb,
            #                                dq.target_Q: targets_mb,
            #                                dq.actions_: actions_mb})

    print("episode is : ", episode)
    print("best game : ", highest_record)
    plt.plot(total_reward)
    plt.plot(total_mean_reward)
    plt.plot(list_20_ep_mean)
    plt.ylabel('episode\'s reward')
    plt.show()

# while i<20:
#     # ball.draw()
#   tk.update_idletasks()
#   tk.update()
#   tk.ai_key_down("'a'")
#   tk.ai_key_down("'s'")
#   tk.ai_key_down("'w'")
#   # while i < 50:
#   i += 1
#   time.sleep(0.1)


while True:
  tk.update_idletasks()
  tk.update()
