# Import modules 
import sys 
import pygame
import cv2
import random 
import numpy as np 
import copy 
import matplotlib.pyplot as plt 
import datetime 
import time 

# Import game
import sys
sys.path.append("Wrapped_Game/")
import dot_Q as game

# Get parameters
Num_action = game.Return_Num_Action()
game_name = game.ReturnName()

# Initial parameters
Replay_memory = []

Num_Exploration = 10000
Num_Training = 20000
Num_Testing  = 5000 

learning_rate = 0.1
gamma = 0.99
first_epsilon = 1.0
final_epsilon = 0.1

step = 1
score = 0 
episode = 0

data_time = str(datetime.date.today()) + '_' + str(datetime.datetime.now().hour) + '_' + str(datetime.datetime.now().minute)

game_state = game.GameState()

# state: my position, enemy position, food position
# action: 0 = south, 1 = north, 2 = East, 3 = West
# reward: eat food = +1, killed by enemy = -1

# Initial state
state =((0,2), (4,2), (2,2))

# Empty Q table
Q_table = {}

while True:
	if step <= Num_Exploration:
		progress = 'Exploration'

		action = np.zeros([Num_action])
		action_index = random.randint(0, Num_action-1)
		action[action_index] = 1
		
		next_state, reward, terminal = game_state.frame_step(action)

		epsilon = first_epsilon

	elif step <= Num_Exploration + Num_Training:
		progress = 'Training'

		# Choose action
		if random.random() < epsilon:
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

		if epsilon > final_epsilon:
    			epsilon -= first_epsilon/Num_Training
		
	elif step <= Num_Exploration + Num_Training + Num_Testing:
		progress = 'Testing'

		# Choose greedy action
		action = np.zeros([Num_action])
		action_index = np.argmax(Q_table[state])
		action[action_index] = 1

		next_state, reward, terminal = game_state.frame_step(action)
		
		epsilon = 0

	else:
		# Finished!
		print('Finished!')
		break

	if state in Q_table.keys() and next_state in Q_table.keys():
		if terminal == True:
			Q_table[state][action_index] = (1 - learning_rate) * Q_table[state][action_index] + learning_rate * (reward)
		else:
			Q_table[state][action_index] = (1 - learning_rate) * Q_table[state][action_index] + learning_rate * (reward + gamma * max(Q_table[next_state]))

	elif state not in Q_table.keys():
		Q_table[state] = []
		for i in range(Num_action):
			Q_table[state].append(0)
	else:
		Q_table[next_state] = []
		for i in range(Num_action):
			Q_table[next_state].append(0)			
	
	# Show progress
	print('Step: ' + str(step) + ' / ' + 'Progress: ' + progress + ' / ' + 'Epsilon: ' + str(epsilon))

	state = next_state
	step += 1