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

# Information
algorithm = 'Q-learning'

# Get parameters
Num_action = game.Return_Num_Action()
game_name = game.ReturnName()

# Initial parameters
Num_Training = 100000
Num_Testing  = 200 

learning_rate = 0.01
gamma = 0.99
first_epsilon = 1.0
final_epsilon = 0.1

epsilon = first_epsilon

Num_plot_episode = 200

step = 1
score = 0 
episode = 1

# Lists for plotting episode - average score
plot_x = []
plot_y = []

# date - hour - minute of training time
date_time = str(datetime.date.today()) + '_' + str(datetime.datetime.now().hour) + '_' + str(datetime.datetime.now().minute)

game_state = game.GameState()

# state: my position, enemy position, food position
# action: 0 = south, 1 = north, 2 = East, 3 = West
# reward: eat food = +1, killed by enemy = -1

# Initial state
state, _, _ = game_state.frame_step(np.zeros([Num_action]))

# Empty Q table
Q_table = {}

while True:
	if step <= Num_Training:
		progress = 'Training'

		# Choose action
		if random.random() < epsilon or state not in Q_table.keys() :
			# Choose random action
			action = np.zeros([Num_action])
			action_index = random.randint(0, Num_action-1)
			action[action_index] = 1
		else:
			# Choose greedy action
			action = np.zeros([Num_action])
			action_index = np.argmax(Q_table[state])
			action[action_index] = 1

		next_state, reward, terminal = game_state.frame_step(action)

		# Update Q-table!
		if state in Q_table.keys() and next_state in Q_table.keys():
			if terminal == True:
				Q_table[state][action_index] = (1 - learning_rate) * Q_table[state][action_index] + learning_rate * (reward)
			else:
				Q_table[state][action_index] = (1 - learning_rate) * Q_table[state][action_index] + learning_rate * (reward + gamma * max(Q_table[next_state]))

		# Decrease epsilon while training
		if epsilon > final_epsilon:
    			epsilon -= first_epsilon/Num_Training
		
	elif step <= Num_Training + Num_Testing:
		progress = 'Testing'

		# Choose greedy action
		action = np.zeros([Num_action])
		action_index = np.argmax(Q_table[state])
		action[action_index] = 1

		next_state, reward, terminal = game_state.frame_step(action)
		epsilon = 0

		# Delay for visualization
		time.sleep(0.25)

	else:
		# Finished!
		print('Finished!')
		plt.savefig('./Plot/' + date_time + '_' + algorithm + '_' + game_name + '.png')
		break

	# If state or next state is not in Q-table, then add it with zeros
	if state not in Q_table.keys():
		Q_table[state] = []
		for i in range(Num_action):
			Q_table[state].append(0)
	elif next_state not in Q_table.keys():
		Q_table[next_state] = []
		for i in range(Num_action):
			Q_table[next_state].append(0)			

	state = next_state
	score += reward
	step += 1

	# Plotting episode - average score
	if len(plot_x) % Num_plot_episode == 0 and len(plot_x) != 0:
		plt.xlabel('Episode')
		plt.ylabel('Score')
		plt.title(algorithm)
		plt.grid(True)

		plt.plot(np.average(plot_x), np.average(plot_y), hold = True, marker = '*', ms = 5)
		plt.draw()
		plt.pause(0.000001)

		plot_x = []
		plot_y = [] 

	# If terminal
	if terminal == True:
		# Show Progress
		print('Step: ' + str(step) + ' / ' + 'Episode: ' + str(episode) + ' / ' + 'Progress: ' + progress + ' / ' + 'Epsilon: ' + str(epsilon) + ' / ' + 'Score: ' + str(score))

		plot_x.append(episode)
		plot_y.append(score)
		episode += 1
		score = 0

		# If game is finished, initialize the state
		state, _, _ = game_state.frame_step(np.zeros([Num_action]))