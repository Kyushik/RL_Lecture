# Deep Q-Network Algorithm

# Import modules
import tensorflow as tf
import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import cv2

# Import game
import sys
sys.path.append("DQN_GAMES/")

import pong as game
import dot
import dot_test
import tetris
import wormy
import breakout

class DQN_Basic:
	def __init__(self):

		# Game Information
		self.algorithm = 'DQN_Basic'
		self.game_name = game.ReturnName()

		# Get parameters
		self.progress = ''
		self.Num_action = game.Return_Num_Action()

		# Initial parameters
		self.Num_Training = 500000
		self.Num_Testing  = 100000

		self.learning_rate = 0.00025
		self.gamma = 0.99

		self.first_epsilon = 1.0
		self.final_epsilon = 0.1

		self.epsilon = self.first_epsilon

		self.Num_plot_episode = 50

		self.step = 1
		self.score = 0
		self.episode = 1

		# Lists for plotting episode - average score
		self.plot_x = []
		self.plot_y = []

		# Save test score
		self.test_score = []

		# date - hour - minute of training time
		self.date_time = str(datetime.date.today()) + '_' + \
		                 str(datetime.datetime.now().hour) + '_' + \
						 str(datetime.datetime.now().minute)

		# Parameters for network
		self.img_size = 80
		self.Num_colorChannel = 1

		self.first_conv   = [8,8,self.Num_colorChannel,32]
		self.second_conv  = [4,4,32,64]
		self.third_conv   = [3,3,64,64]
		self.first_dense  = [10*10*64, 512]
		self.second_dense = [512, self.Num_action]

		self.GPU_fraction = 0.2

		# Initialize Network
		self.input, self.output = self.network()
		self.train_step, self.action_target, self.y_target = self.loss_and_train()
		self.init_sess()

	def main(self):
		# Define game state
		game_state = game.GameState()

		# Initialization
		action = np.zeros([self.Num_action])
		state, _, _ = game_state.frame_step(action)
		state = self.reshape_input(state)

		while True:
			# Get progress:
			#	Training: Linearly decease epsilon
			#	Testing : Epsilon = 0
			self.progress = self.get_progress()

			# Select action
			action = self.select_action(state)

			# Take action and get info. for update
			next_state, reward, terminal = game_state.frame_step(action)
			next_state = self.reshape_input(next_state)

			# Training the Q-table!
			self.train(state, action, reward, next_state, terminal)

			# Plotting
			self.plotting()

			# Update former info.
			state = next_state
			self.score += reward
			self.step += 1

			# If game is over (terminal)
			if terminal:
				state = self.if_terminal(game_state)

			# Finished!
			if self.progress == 'Finished':
				print('Finished!')

				avg_test_score = str(sum(self.test_score) / len(self.test_score))
				print('Average Test score: ' + avg_test_score)
				plt.savefig('./Plot/' + self.date_time + '_' +self.algorithm + '_' + self.game_name + avg_test_score + '.png')
				break

	def init_sess(self):
		# Initialize variables
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = self.GPU_fraction

		sess = tf.InteractiveSession(config=config)
		init = tf.global_variables_initializer()
		sess.run(init)

	def get_progress(self):
		progress = ''
		if self.step <= self.Num_Training:
			progress = 'Training'
		elif self.step <= self.Num_Training + self.Num_Testing:
			progress = 'Testing'
		else:
			progress = 'Finished'

		return progress

	# Resize and make input as grayscale
	def reshape_input(self, state):
		state_out = cv2.resize(state, (self.img_size, self.img_size))
		if self.Num_colorChannel == 1:
			state_out = cv2.cvtColor(state_out, cv2.COLOR_BGR2GRAY)
			state_out = np.reshape(state_out, (self.img_size, self.img_size, 1))

		state_out = np.uint8(state_out)
		return state_out

	# Convolution and pooling
	def conv2d(self, x, w, stride):
		return tf.nn.conv2d(x,w,strides=[1, stride, stride, 1], padding='SAME')

	# Get Variables
	def conv_weight_variable(self, name, shape):
	    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer_conv2d())

	def weight_variable(self, name, shape):
	    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

	def bias_variable(self, name, shape):
	    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

	def network(self):
		# Input
		x_image = tf.placeholder(tf.float32, shape = [None,self.img_size, self.img_size, self.Num_colorChannel])
		x_normalize = (x_image - (255.0/2)) / (255.0/2)

		with tf.variable_scope('network'):
			# Convolution variables
			w_conv1 = self.conv_weight_variable('w_conv1', self.first_conv)
			b_conv1 = self.bias_variable('b_conv1',[self.first_conv[3]])

			w_conv2 = self.conv_weight_variable('w_conv2',self.second_conv)
			b_conv2 = self.bias_variable('b_conv2',[self.second_conv[3]])

			w_conv3 = self.conv_weight_variable('w_conv3',self.third_conv)
			b_conv3 = self.bias_variable('b_conv3',[self.third_conv[3]])

			# Densely connect layer variables
			w_fc1 = self.weight_variable('w_fc1',self.first_dense)
			b_fc1 = self.bias_variable('b_fc1',[self.first_dense[1]])

			w_fc2 = self.weight_variable('w_fc2',self.second_dense)
			b_fc2 = self.bias_variable('b_fc2',[self.second_dense[1]])

		# Network
		h_conv1 = tf.nn.relu(self.conv2d(x_normalize, w_conv1, 4) + b_conv1)
		h_conv2 = tf.nn.relu(self.conv2d(h_conv1, w_conv2, 2) + b_conv2)
		h_conv3 = tf.nn.relu(self.conv2d(h_conv2, w_conv3, 1) + b_conv3)

		h_pool3_flat = tf.reshape(h_conv3, [-1, self.first_dense[0]])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1)+b_fc1)

		output = tf.matmul(h_fc1, w_fc2) + b_fc2

		return x_image, output

	def loss_and_train(self):
		# Loss function and Train
		action_target = tf.placeholder(tf.float32, shape = [None, self.Num_action])
		y_target = tf.placeholder(tf.float32, shape = [None])

		y_prediction= tf.reduce_sum(tf.multiply(self.output, action_target), reduction_indices = 1)
		Loss = tf.reduce_mean(tf.square(y_prediction - y_target))
		train_step = tf.train.AdamOptimizer(learning_rate = self.learning_rate, epsilon = 1e-02).minimize(Loss)

		return train_step, action_target, y_target

	def select_action(self, state):
		action = np.zeros([self.Num_action])
		action_index = 0

		# Choose action
		if self.progress == 'Training':
			if random.random() < self.epsilon:
				# Choose random action
				action_index = random.randint(0, self.Num_action-1)
				action[action_index] = 1
			else:
				# Choose greedy action
				Q_value = self.output.eval(feed_dict={self.input: [state]})
				action_index = np.argmax(Q_value)
				action[action_index] = 1

			# Decrease epsilon while training
			if self.epsilon > self.final_epsilon:
				self.epsilon -= self.first_epsilon/self.Num_Training

		elif self.progress == 'Testing':
			# Choose greedy action
			Q_value = self.output.eval(feed_dict={self.input: [state]})
			action_index = np.argmax(Q_value)
			action[action_index] = 1

			self.epsilon = 0

		return action

	def train(self, state, action, reward, next_state, terminal):
		y = []
		Q = self.output.eval(feed_dict = {self.input: [next_state]})

		if terminal == True:
			y.append(reward)
		else:
			y.append(reward + self.gamma * np.max(Q))

		self.train_step.run(feed_dict = {self.action_target: [action],
										 self.y_target: y,
										 self.input: [state]})

	def plotting(self):
		# Plotting episode - average score
		if len(self.plot_x) % self.Num_plot_episode == 0 and len(self.plot_x) != 0:
			plt.xlabel('Episode')
			plt.ylabel('Score')
			plt.title(self.algorithm)
			plt.grid(True)

			plt.plot(np.average(self.plot_x), np.average(self.plot_y), hold = True, marker = '*', ms = 5)
			plt.draw()
			plt.pause(0.000001)

			self.plot_x = []
			self.plot_y = []

	def if_terminal(self, game_state):
		# Show Progress
		print('Step: ' + str(self.step) + ' / ' +
		      'Episode: ' + str(self.episode) + ' / ' +
			  'Progress: ' + self.progress + ' / ' +
			  'Epsilon: ' + str(self.epsilon) + ' / ' +
			  'Score: ' + str(self.score))

		if self.progress == 'Testing':
			self.test_score.append(self.score)

		self.plot_x.append(self.episode)
		self.plot_y.append(self.score)
		self.episode += 1
		self.score = 0

		# If game is finished, initialize the state
		state, _, _ = game_state.frame_step(np.zeros([self.Num_action]))
		state = self.reshape_input(state)

		return state

if __name__ == '__main__':
	agent = DQN_Basic()
	agent.main()
