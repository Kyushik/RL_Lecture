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

class DQN_Final:
	def __init__(self):

		# Game Information
		self.algorithm = 'DQN_Final'
		self.game_name = game.ReturnName()

		# Get parameters
		self.progress = ''
		self.Num_action = game.Return_Num_Action()

		# Initial parameters
		self.Num_Exploration = 50000
		self.Num_Training    = 500000
		self.Num_Testing     = 100000

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

		# parameters for skipping and stacking
		self.state_set = []
		self.Num_skipping = 4
		self.Num_stacking = 4

		# Parameter for Experience Replay
		self.Num_replay_memory = 50000
		self.Num_batch = 32
		self.replay_memory = []

		# Parameter for Target Network
		self.Num_update_target = 5000

		# Parameters for network
		self.img_size = 80
		self.Num_colorChannel = 1

		self.first_conv   = [8,8,self.Num_stacking * self.Num_colorChannel,32]
		self.second_conv  = [4,4,32,64]
		self.third_conv   = [3,3,64,64]
		self.first_dense  = [10*10*64, 512]
		self.second_dense = [512, self.Num_action]

		self.GPU_fraction = 0.2

		# Initialize Network
		self.input, self.output = self.network()
		self.input_target, self.output_target = self.target_network()
		self.train_step, self.action_target, self.y_prediction = self.loss_and_train()
		self.sess = self.init_sess()

	def main(self):
		# Define game state
		game_state = game.GameState()

		# Initialization
		state = self.initialization(game_state)
		stacked_state = self.skip_and_stack_frame(state)

		while True:
			# Get progress:
			self.progress = self.get_progress()

			# Select action
			action = self.select_action(stacked_state)

			# Take action and get info. for update
			next_state, reward, terminal = game_state.frame_step(action)
			next_state = self.reshape_input(next_state)
			stacked_next_state = self.skip_and_stack_frame(next_state)

			# Experience Replay
			self.experience_replay(stacked_state, action, reward, stacked_next_state, terminal)

			# Training!
			if self.progress == 'Training':
				# Update target network
				if self.step % self.Num_update_target == 0:
					self.update_target()

				# Training
				self.train(self.replay_memory)

			# Plotting
			self.plotting()

			# Update former info.
			stacked_state = stacked_next_state
			self.score += reward
			self.step += 1

			# If game is over (terminal)
			if terminal:
				stacked_state = self.if_terminal(game_state)

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

		return sess

	def initialization(self, game_state):
		action = np.zeros([self.Num_action])
		state, _, _ = game_state.frame_step(action)
		state = self.reshape_input(state)

		for i in range(self.Num_skipping * self.Num_stacking):
			self.state_set.append(state)

		return state

	def skip_and_stack_frame(self, state):
		self.state_set.append(state)

		state_in = np.zeros((self.img_size, self.img_size, self.Num_colorChannel * self.Num_stacking))

		# Stack the frame according to the number of skipping frame
		for stack_frame in range(self.Num_stacking):
			state_in[:,:,stack_frame] = self.state_set[-1 - (self.Num_skipping * stack_frame)]

		del self.state_set[0]

		state_in = np.uint8(state_in)
		return state_in

	def get_progress(self):
		progress = ''
		if self.step <= self.Num_Exploration:
			progress = 'Exploring'
		elif self.step <= self.Num_Exploration + self.Num_Training:
			progress = 'Training'
		elif self.step <= self.Num_Exploration + self.Num_Training + self.Num_Testing:
			progress = 'Testing'
		else:
			progress = 'Finished'

		return progress

	# Resize and make input as grayscale
	def reshape_input(self, state):
		state_out = cv2.resize(state, (self.img_size, self.img_size))
		if self.Num_colorChannel == 1:
			state_out = cv2.cvtColor(state_out, cv2.COLOR_BGR2GRAY)
			state_out = np.reshape(state_out, (self.img_size, self.img_size))

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
		x_image = tf.placeholder(tf.float32, shape = [None,
													  self.img_size,
													  self.img_size,
													  self.Num_stacking * self.Num_colorChannel])

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

	def target_network(self):
		# Input
		x_image_target = tf.placeholder(tf.float32, shape = [None,
													  		self.img_size,
													  		self.img_size,
													  		self.Num_stacking * self.Num_colorChannel])

		x_normalize_target = (x_image_target - (255.0/2)) / (255.0/2)

		with tf.variable_scope('target'):
			# Convolution variables
			w_conv1_target = self.conv_weight_variable('w_conv1', self.first_conv)
			b_conv1_target = self.bias_variable('b_conv1',[self.first_conv[3]])

			w_conv2_target = self.conv_weight_variable('w_conv2',self.second_conv)
			b_conv2_target = self.bias_variable('b_conv2',[self.second_conv[3]])

			w_conv3_target = self.conv_weight_variable('w_conv3',self.third_conv)
			b_conv3_target = self.bias_variable('b_conv3',[self.third_conv[3]])

			# Densely connect layer variables
			w_fc1_target = self.weight_variable('w_fc1',self.first_dense)
			b_fc1_target = self.bias_variable('b_fc1',[self.first_dense[1]])

			w_fc2_target = self.weight_variable('w_fc2',self.second_dense)
			b_fc2_target = self.bias_variable('b_fc2',[self.second_dense[1]])

		# Network
		h_conv1_target = tf.nn.relu(self.conv2d(x_normalize_target, w_conv1_target, 4) + b_conv1_target)
		h_conv2_target = tf.nn.relu(self.conv2d(h_conv1_target, w_conv2_target, 2) + b_conv2_target)
		h_conv3_target = tf.nn.relu(self.conv2d(h_conv2_target, w_conv3_target, 1) + b_conv3_target)

		h_pool3_flat_target = tf.reshape(h_conv3_target, [-1, self.first_dense[0]])
		h_fc1_target = tf.nn.relu(tf.matmul(h_pool3_flat_target, w_fc1_target)+b_fc1_target)

		output_target = tf.matmul(h_fc1_target, w_fc2_target) + b_fc2_target
		return x_image_target, output_target

	def loss_and_train(self):
		# Loss function and Train
		action_target = tf.placeholder(tf.float32, shape = [None, self.Num_action])
		y_prediction = tf.placeholder(tf.float32, shape = [None])

		y_target = tf.reduce_sum(tf.multiply(self.output, action_target), reduction_indices = 1)
		Loss = tf.reduce_mean(tf.square(y_prediction - y_target))
		train_step = tf.train.AdamOptimizer(learning_rate = self.learning_rate, epsilon = 1e-02).minimize(Loss)

		return train_step, action_target, y_prediction

	def select_action(self, stacked_state):
		action = np.zeros([self.Num_action])
		action_index = 0

		# Choose action
		if self.progress == 'Exploring':
			# Choose random action
			action_index = random.randint(0, self.Num_action-1)
			action[action_index] = 1

		elif self.progress == 'Training':
			if random.random() < self.epsilon:
				# Choose random action
				action_index = random.randint(0, self.Num_action-1)
				action[action_index] = 1
			else:
				# Choose greedy action
				Q_value = self.output.eval(feed_dict={self.input: [stacked_state]})
				action_index = np.argmax(Q_value)
				action[action_index] = 1

			# Decrease epsilon while training
			if self.epsilon > self.final_epsilon:
				self.epsilon -= self.first_epsilon/self.Num_Training

		elif self.progress == 'Testing':
			# Choose greedy action
			Q_value = self.output.eval(feed_dict={self.input: [stacked_state]})
			action_index = np.argmax(Q_value)
			action[action_index] = 1

			self.epsilon = 0

		return action

	def experience_replay(self, state, action, reward, next_state, terminal):
		# If Replay memory is longer than Num_replay_memory, delete the oldest one
		if len(self.replay_memory) > self.Num_replay_memory:
			del self.replay_memory[0]
		else:
			self.replay_memory.append([state, action, reward, next_state, terminal])

	def update_target(self):
		# Get trainable variables
		trainable_variables = tf.trainable_variables()
		# network variables
		trainable_variables_network = [var for var in trainable_variables if var.name.startswith('network')]

		# target variables
		trainable_variables_target = [var for var in trainable_variables if var.name.startswith('target')]

		for i in range(len(trainable_variables_network)):
			self.sess.run(tf.assign(trainable_variables_target[i], trainable_variables_network[i]))

	def train(self, replay_memory):
		# Select minibatch
		minibatch =  random.sample(replay_memory, self.Num_batch)

		# Save the each batch data
		state_batch      = [batch[0] for batch in minibatch]
		action_batch     = [batch[1] for batch in minibatch]
		reward_batch     = [batch[2] for batch in minibatch]
		next_state_batch = [batch[3] for batch in minibatch]
		terminal_batch   = [batch[4] for batch in minibatch]

		# Get y_prediction
		y_batch = []
		Q_batch = self.output_target.eval(feed_dict = {self.input_target: next_state_batch})

		# Get target values
		for i in range(len(minibatch)):
			if terminal_batch[i] == True:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + self.gamma * np.max(Q_batch[i]))

		self.train_step.run(feed_dict = {self.action_target: action_batch,
										 self.y_prediction: y_batch,
										 self.input: state_batch})

	def plotting(self):
		# Plotting episode - average score
		if len(self.plot_x) % self.Num_plot_episode == 0 and len(self.plot_x) != 0 and self.progress != 'Exploring':
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

		if self.progress != 'Exploring':
			self.plot_x.append(self.episode)
			self.plot_y.append(self.score)
			self.episode += 1

		self.score = 0

		# If game is finished, initialize the state
		state = self.initialization(game_state)
		stacked_state = self.skip_and_stack_frame(state)

		return stacked_state

if __name__ == '__main__':
	agent = DQN_Final()
	agent.main()
