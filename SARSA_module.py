# SARSA algorithm

# ---------------------------------------------------
# Game Information
# state: my position, enemy position, food position
# action: 0 = south, 1 = north, 2 = East, 3 = West
# reward: eat food = +1, killed by enemy = -1

# Import modules
import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

# Import game
import sys
sys.path.append("DQN_GAMES/")

import dot_Q as game

class SARSA:
	def __init__(self):

		# Game Information
		self.algorithm = 'SARSA'
		self.game_name = game.ReturnName()

		# Get parameters
		self.progress = ''
		self.Num_action = game.Return_Num_Action()

		# Initial parameters
		self.Num_Training = 100000
		self.Num_Testing  = 1000

		self.learning_rate = 0.01
		self.gamma = 0.99

		self.first_epsilon = 1.0
		self.final_epsilon = 0.1

		self.epsilon = self.first_epsilon

		self.Num_plot_episode = 200

		self.step = 1
		self.score = 0
		self.episode = 1

		# Lists for plotting episode - average score
		self.plot_x = []
		self.plot_y = []

		# date - hour - minute of training time
		self.date_time = str(datetime.date.today()) + '_' + str(datetime.datetime.now().hour) + '_' + str(datetime.datetime.now().minute)

		# Empty Q table
		self.Q_table = {}

	def main(self):
		# Define game state
		game_state = game.GameState()

		# Initial state
		state, _, _ = game_state.frame_step(np.zeros([self.Num_action]))

		# Initial action
		action = np.zeros([self.Num_action])
		action_index = random.randint(0, self.Num_action-1)
		action[action_index] = 1

		while True:
			# Get progress:
			#   Training: linearly decrease Epsilon
			#   Testing:  epsilon = 0
			self.progress = self.get_progress()

			# Take action and get info. for update
			next_state, reward, terminal = game_state.frame_step(action)

			# Select action: 0 = south, 1 = north, 2 = East, 3 = West
			next_action = self.select_action(next_state)

			# Decrease epsilon for epsilon greedy
			self.decrease_epsilon()

			# Training the Q-table!
			self.train(state, action, reward, next_state, next_action, terminal)

			# Delay for visualization
			if self.progress == 'Testing':
				time.sleep(0.25)

			# Plotting
			self.plotting()

			# Finished!
			if self.progress == 'Finished':
				print('Finished!')
				plt.savefig('./Plot/' + self.date_time + '_' +self.algorithm + '_' + self.game_name + '.png')
				break

			# Update former info.
			state = next_state
			action = next_action
			self.score += reward
			self.step += 1

			# If game is over (terminal)
			if terminal:
				state, action = self.if_terminal(game_state)

	def get_progress(self):
		progress = ''
		if self.step <= self.Num_Training:
			progress = 'Training'
		elif self.step <= self.Num_Training + self.Num_Testing:
			progress = 'Testing'
		else:
			progress = 'Finished'

		return progress

	def select_action(self, state):
		action = np.zeros([self.Num_action])
		action_index = 0

		# Choose action
		if self.progress == 'Training':
			if random.random() < self.epsilon or state not in self.Q_table.keys() :
				# Choose random action
				action_index = random.randint(0, self.Num_action-1)
				action[action_index] = 1
			else:
				# Choose greedy action
				action_index = np.argmax(self.Q_table[state])
				action[action_index] = 1
		elif self.progress == 'Testing':
			# Choose greedy action
			action_index = np.argmax(self.Q_table[state])
			action[action_index] = 1

		return action

	def decrease_epsilon(self):
		if self.progress == 'Training':
			# Decrease epsilon while training
			if self.epsilon > self.final_epsilon:
				self.epsilon -= self.first_epsilon/self.Num_Training
		else:
			self.epsilon = 0

	def train(self, state, action, reward, next_state, next_action, terminal):
		# If state or next state is not in Q-table, then add it with zeros
		if state not in self.Q_table.keys():
			self.Q_table[state] = []
			for i in range(self.Num_action):
				self.Q_table[state].append(0)
		if next_state not in self.Q_table.keys():
			self.Q_table[next_state] = []
			for i in range(self.Num_action):
				self.Q_table[next_state].append(0)

		action_index = np.argmax(action)
		next_action_index = np.argmax(next_action)
		# Update Q-table!
		if state in self.Q_table.keys() and next_state in self.Q_table.keys():
			if terminal == True:
				self.Q_table[state][action_index] = (1 - self.learning_rate) * self.Q_table[state][action_index] + self.learning_rate * (reward)
			else:
				self.Q_table[state][action_index] = (1 - self.learning_rate) * self.Q_table[state][action_index] + self.learning_rate * (reward + self.gamma * self.Q_table[next_state][next_action_index])

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

		self.plot_x.append(self.episode)
		self.plot_y.append(self.score)
		self.episode += 1
		self.score = 0

		# If game is finished, initialize the state
		action = np.zeros([self.Num_action])
		state, _, _ = game_state.frame_step(action)

		return state, action

if __name__ == '__main__':
	agent = SARSA()
	agent.main()
