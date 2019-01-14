#! -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, random, time
from collections import deque

from PyPong import PyPong

weight_path = "model/dqn_cartpole_weights"
load_weights = False
skip_training = False

# Parameters.
alpha = 0.0001		# Learning rate.
gamma = 0.99		# Future reward discount rate.

# Exploration / Exploitation rate.
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999

batch_size = 64
max_episode = 500
max_timestep = 250.0
memory_max_size = 1000000
last_reward_record_count = 100

max_timestep_increment_after_episode = 100
max_timestep_increment = 10

env = PyPong(clock_tick=60)
observation = env.reset()

obvSpace_dim = env.observation_space_n
actSpace_dim = env.action_space_n

# Define Neural Network.
sess = tf.Session()

# Neural network config.
agent1_layer_cfg = [
	[32,			tf.nn.relu],
	[32,			tf.nn.relu],
	[actSpace_dim,	None]
]

agent2_layer_cfg = [
	[32,			tf.nn.relu],
	[32,			tf.nn.relu],
	[actSpace_dim,	None]
]

# Agent class.
class DQNAgent:
	def __init__(self, name, layer_cfg):
		with tf.variable_scope(name):
			# Create dense layers and store them in a list.
			layers = [
				tf.layers.Dense(
					units=layer_cfg[l][0],
					activation=layer_cfg[l][1],
					kernel_initializer=(
						lambda shape, dtype, partition_info: tf.Variable(tf.random_uniform(shape, -1.0, 1.0, dtype=dtype))
					)
				)
				for l in range(len(layer_cfg))
			]

			# Feed forward given matrix to the model.
			def feedForward(x):
				for layer in layers:
					x = layer(x)
				return x

			# Inputs.
			self.tf_state = tf.placeholder(tf.float32, shape=(None, obvSpace_dim))
			self.tf_action = tf.placeholder(tf.float32, shape=(None, actSpace_dim))

			# Feed-forward.
			self.tf_output = feedForward(self.tf_state)
			self.tf_q_pred = tf.reduce_sum(
				tf.multiply(self.tf_output, self.tf_action),
				axis=1
			)

			# Feed-backward.
			self.tf_q_tar = tf.placeholder(tf.float32, shape=(None))

			self.tf_loss = tf.reduce_mean(tf.square(self.tf_q_tar - self.tf_q_pred))
			self.tf_train = tf.train.GradientDescentOptimizer(alpha).minimize(self.tf_loss)

# Copies all trainable parameters from model mName to fName.
# This function gonna be called at every end of the episode.
# mName: Model to be trained.
# fName: Fixed model.
def update_parameters(mName, fName):
	updateOp = []
	for mW, fW in zip(
			tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, mName),
			tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, fName)
		):
		updateOp.append(fW.assign(mW))
	sess.run(updateOp)

agent1_model = DQNAgent("agent1_model", agent1_layer_cfg)				# Agent's DQN model.
agent1_fixed_model = DQNAgent("agent1_fixed_model", agent1_layer_cfg)	# Fixed weights model, only used for predicting 'next_state'. Q(s', a')

agent2_model = DQNAgent("agent2_model", agent2_layer_cfg)				# Agent's DQN model.
agent2_fixed_model = DQNAgent("agent2_fixed_model", agent2_layer_cfg)	# Fixed weights model, only used for predicting 'next_state'. Q(s', a')

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# Try to load weights.
if load_weights:
	try:
		saver.restore(sess, weight_path)
		print("Weights loaded.")

		if skip_training:
			print("Skipping training.")
	except:
		skip_training = False
		print("Weights couldn't loaded.")
else:
	skip_training = False

# Our agent's memory.
# Going to store experiences in [State, Action, Reward, NextState] form.
agent1_Memory = deque(maxlen=memory_max_size)
agent2_Memory = deque(maxlen=memory_max_size)

# Training.
plot_reward_record = []
reward_record = []
for episode in range(max_episode):
	if skip_training:
		break

	agent1_state, agent2_state = env.reset()
	episode_reward = [0.0, 0.0]

	for timestep in range(int(max_timestep)):
		# Comment this if you don't want to see the simulation while agent learns. (Slows training!)
		# env.render()

		# Explore / Exploit decision.
		# Based on epsilon value which is between 0 and 1.

		# Agent1
		if random.random() < epsilon:
			# Explore, act randomly.
			agent1_action = random.randint(0, actSpace_dim-1)
		else:
			# Exploit, act the action which gives most reward.
			agent1_action = np.argmax(
				sess.run(
					agent1_model.tf_output,
					feed_dict={
						agent1_model.tf_state:np.expand_dims(agent1_state, axis=0)
					}
				)[0]
			)

		# Agent2
		if random.random() < epsilon:
			# Explore, act randomly.
			agent2_action = random.randint(0, actSpace_dim-1)
		else:
			# Exploit, act the action which gives most reward.
			agent2_action = np.argmax(
				sess.run(
					agent2_model.tf_output,
					feed_dict={
						agent2_model.tf_state:np.expand_dims(agent2_state, axis=0)
					}
				)[0]
			)

		# Apply action on simulation.
		[agent1_next_state, agent2_next_state], [agent1_reward, agent2_reward], done, info = env.step([agent1_action, agent2_action])

		episode_reward[0] = episode_reward[0] + agent1_reward
		episode_reward[1] = episode_reward[1] + agent2_reward

		# Store experiences in memories. Only do this if it's not the end step.
		if not done:
			agent1_Memory.append(
				[agent1_state, agent1_action, agent1_reward, agent1_next_state]
			)
			agent2_Memory.append(
				[agent2_state, agent2_action, agent2_reward, agent2_next_state]
			)

		agent1_state = agent1_next_state
		agent2_state = agent2_next_state

		# Experience replay (training).
		# Waits until experiences accumulate much as batch size.

		# Train Agent1
		if len(agent1_Memory) > batch_size:
			mini_batch = random.sample(agent1_Memory, batch_size)
			
			a1_b_actions = np.array([np.array(b[1]) for b in mini_batch])
			a1_b_actions_o = np.eye(actSpace_dim)[a1_b_actions].astype(np.float32)
			a1_b_states = np.array([np.array(b[0]) for b in mini_batch])

			# Get fixed model's output. And train the other model with this output.
			a1_b_next_states = np.array([np.array(b[3]) for b in mini_batch])
			a1_b_rewards = np.array([np.array(b[2]) for b in mini_batch])

			# Get the next state's most rewarding action for each experience.
			a1_b_q_tars = a1_b_rewards + gamma * np.max(
				sess.run(
					agent1_fixed_model.tf_output,
					feed_dict={
						agent1_fixed_model.tf_state:a1_b_next_states
					}
				),
				axis=1
			)

			a1_feed = {
				agent1_model.tf_action:a1_b_actions_o,
				agent1_model.tf_state:a1_b_states,
				agent1_model.tf_q_tar:a1_b_q_tars
			}

			# Finally, train the model!
			sess.run(
				agent1_model.tf_train,
				feed_dict=a1_feed
			)

		# Train Agent2
		if len(agent2_Memory) > batch_size:
			mini_batch = random.sample(agent2_Memory, batch_size)
			
			a2_b_actions = np.array([np.array(b[1]) for b in mini_batch])
			a2_b_actions_o = np.eye(actSpace_dim)[a2_b_actions].astype(np.float32)
			a2_b_states = np.array([np.array(b[0]) for b in mini_batch])

			# Get fixed model's output. And train the other model with this output.
			a2_b_next_states = np.array([np.array(b[3]) for b in mini_batch])
			a2_b_rewards = np.array([np.array(b[2]) for b in mini_batch])

			# Get the next state's most rewarding action for each experience.
			a2_b_q_tars = a2_b_rewards + gamma * np.max(
				sess.run(
					agent2_fixed_model.tf_output,
					feed_dict={
						agent2_fixed_model.tf_state:a2_b_next_states
					}
				),
				axis=1
			)

			a2_feed = {
				agent2_model.tf_action:a2_b_actions_o,
				agent2_model.tf_state:a2_b_states,
				agent2_model.tf_q_tar:a2_b_q_tars
			}

			# Finally, train the model!
			sess.run(
				agent2_model.tf_train,
				feed_dict=a2_feed
			)

		if epsilon > epsilon_min:
			epsilon *= epsilon_decay

		if done or (timestep+1 == int(max_timestep)):
			if len(reward_record) > last_reward_record_count:
				reward_record = reward_record[-last_reward_record_count:]

			if len(reward_record) > 0:
				avgRew = np.mean(np.array(reward_record), axis=0)
			else:
				avgRew = np.array([0.0, 0.0])

			if episode > max_timestep_increment_after_episode:
				max_timestep += max_timestep_increment

			print(
				"Ep %s, Epsilon %s, Ep Reward %s, Loss %s, Max Step %s, Avg reward %s" % (
					episode,
					"{0:.2f}".format(epsilon),
					episode_reward,
					([sess.run(agent1_model.tf_loss, feed_dict=a1_feed), sess.run(agent2_model.tf_loss, feed_dict=a2_feed)]
						if len(agent1_Memory) > batch_size else None),
					"{0:.2f}".format(max_timestep),
					["{0:.2f}".format(avgRew[0]), "{0:.2f}".format(avgRew[1])]
				)
			)

			# Copy trained parameters to fixed model every end of the episode.
			update_parameters("agent1_model", "agent1_fixed_model")
			update_parameters("agent2_model", "agent2_fixed_model")

			reward_record.append(episode_reward)
			plot_reward_record.append(episode_reward)
			break

# Plot reward for each episode.
if not skip_training:
	saver.save(sess, weight_path)

	plot_reward_record = np.array([np.array(p) for p in plot_reward_record])

	plt.plot(range(plot_reward_record.shape[0]), plot_reward_record[:, 0])
	plt.plot(range(plot_reward_record.shape[0]), plot_reward_record[:, 1])
	plt.title("DQN Pong Duel Reward")
	plt.show()

# Testing!
agent1_state, agent2_state = env.reset()
while True:
	env.render()

	# Selecting the best action for current state.
	# Agent1
	agent1_action = np.argmax(
		sess.run(
			agent1_model.tf_output,
			feed_dict={
				agent1_model.tf_state:np.expand_dims(agent1_state, axis=0)
			}
		)[0]
	)

	# Agent2
	agent2_action = np.argmax(
		sess.run(
			agent2_model.tf_output,
			feed_dict={
				agent2_model.tf_state:np.expand_dims(agent2_state, axis=0)
			}
		)[0]
	)

	[agent1_state, agent2_state], reward, done, info = env.step([agent1_action, agent2_action])
	
	if done:
		agent1_state, agent2_state = env.reset()