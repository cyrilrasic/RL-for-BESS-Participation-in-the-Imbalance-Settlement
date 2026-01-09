# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:03:27 2024

@author: Cyril

TD3 Main
"""

# %% -------import-------
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

tf.keras.backend.set_floatx('float32')

from tf_agents.agents.td3 import td3_agent
from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common

import time
import os

from classes_TD3_penalized import floor, Environment, Environmentvalidation, importdata, compute_avg_return, \
    collect_data, plotdata

from datetime import datetime

tf.keras.backend.clear_session()

# %% -------Title of run-------
title_of_run = ('Title of run')
print(title_of_run)

# %%
seed = 2
np.random.seed(seed)
tf.random.set_seed(seed)

# %% -------Parameters + Data-------

# battery param
battery_max_power = 20  # MW

EP_ratio = 1

delta_t = 0.25  # time control interval (h)

roundtrip_efficiency = 0.9

battery_replacement_cost = 3e+5 * battery_max_power * EP_ratio

# number of Look Ahead steps
LA_steps = 8

# penalty for physical decisions
penal = 200

# Safe projection mode (turns on the cvxpy if true)
SafeProjectionMode = False

# data path
path_training = os.path.normpath('../../Data/Frame_1819_training.xlsx')
path_validation = os.path.normpath('../../Data/Frame_1819_validation.xlsx')
path_test = os.path.normpath('../../Data/Frame_1819_test.xlsx')

# data import
print('Importing data + preprocessing...')
index_col_si, index_first_col_MDP, scaler, scaler_list, train_observable_samples, validation_observable_samples, test_observable_samples, \
    train_exact_si_samples, validation_exact_si_samples, test_exact_si_samples = importdata(path_training,
                                                                                            path_validation, path_test,
                                                                                            LA_steps)
print('Preproced data imported')

index_soc = 28

# %% -------Hyperparameters-------

tuned_initial_exp = 1  # 0 = yes, 1 = no

# nb of steps per ep
nb_quarters_per_episode = 96

replay_buffer_max_length = int(2 * 1e4)

batch_size = 256

# number of epochs in the training
num_epochs = 0.1

# gamma, discounting rate of futures rewards
# depends on LA_steps, without the info the critic should not compute too far rewards
gamma = np.float64(0.99)

# tau, soft update of the targets
tau = 1e-2

# learning rates
actor_lr = 1e-5
critic_lr = 1e-4

# optimizers
actor_optimizer = tf.keras.optimizers.legacy.Adam(actor_lr, clipnorm=1)
critic_optimizer = tf.keras.optimizers.legacy.Adam(critic_lr, clipnorm=1)

actor_update_T = 5
target_update_T = 5

# architecture, hidden layers of the nn
hidden_actor = (500, 500)
hidden_critic = (500, 500)

# activations
actor_activ = tf.nn.relu
critic_activ = tf.nn.relu

# noise
noise_stddev = 0.2
# number of experiences initially added in the buffer
initial_collect_steps = batch_size * 2

# number of experiences stored at each training iteration in the buffer
collect_steps_per_iteration = 4

num_iterations = int(
    num_epochs * int((len(train_observable_samples) - initial_collect_steps) / collect_steps_per_iteration))

num_training_episodes = int(num_iterations / nb_quarters_per_episode)

log_interval = floor(num_iterations / 2)  # the loss is printed every "log_interval" iterations.

eval_interval = floor(num_iterations / 2)  # the agent is tested every "eval_interval" steps

num_test_episodes = int(len(test_observable_samples) / nb_quarters_per_episode)

num_validation_episodes = int(len(validation_observable_samples) / nb_quarters_per_episode)

# %% -------Environments-------
starting_time = time.time()

# Training env 
train_env = Environment(train_observable_samples, train_exact_si_samples, scaler, scaler_list, battery_max_power,
                        1, nb_quarters_per_episode, index_first_col_MDP, EP_ratio, roundtrip_efficiency,
                        battery_replacement_cost, penalty=penal)

# The python environment we created is wrapped into a tensorflow environment
tf_train_env = tf_py_environment.TFPyEnvironment(train_env)

# Validation env 
validation_env = Environmentvalidation(validation_observable_samples, validation_exact_si_samples, scaler, scaler_list,
                                       battery_max_power,
                                       1, nb_quarters_per_episode, index_first_col_MDP, EP_ratio, roundtrip_efficiency,
                                       battery_replacement_cost)

# The python environment we created is wrapped into a tensorflow environment
tf_validation_env = tf_py_environment.TFPyEnvironment(validation_env)

# Testing env 
test_env = Environmentvalidation(test_observable_samples, test_exact_si_samples, scaler, scaler_list, battery_max_power,
                                 1, nb_quarters_per_episode, index_first_col_MDP, EP_ratio, roundtrip_efficiency,
                                 battery_replacement_cost)

# The python environment we created is wrapped into a tensorflow environment
tf_test_env = tf_py_environment.TFPyEnvironment(test_env)

# %% -------Actor and critic-------
Critic = critic_network.CriticNetwork((tf_train_env.observation_spec(), tf_train_env.action_spec()),
                                      joint_fc_layer_params=hidden_critic,
                                      activation_fn=critic_activ)


BESS_layer_cvxpy = 1
Actor = actor_network.ActorNetwork(tf_train_env.observation_spec(), tf_train_env.action_spec(), hidden_actor,
                                       activation_fn=actor_activ)
target_actor = actor_network.ActorNetwork(tf_train_env.observation_spec(), tf_train_env.action_spec(), hidden_actor,
                                              activation_fn=actor_activ)

# %% -------Agent-------
train_step_counter = tf.Variable(0)
agent = td3_agent.Td3Agent(tf_train_env.time_step_spec(), tf_train_env.action_spec(), Actor, Critic,
                           actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                           exploration_noise_std=noise_stddev,
                           target_update_tau=tau, target_update_period=target_update_T,
                           actor_update_period=actor_update_T,
                           td_errors_loss_fn=None, gamma=gamma, target_policy_noise=1,
                           reward_scale_factor=np.float64(1),
                           target_policy_noise_clip=1, gradient_clipping=None, debug_summaries=None,
                           target_actor_network=target_actor,
                           summarize_grads_and_vars=False, train_step_counter=train_step_counter)

agent.initialize()

# -------policies-------
# the real policy of the agent 
eval_policy = agent.policy

# policy including noise
collect_policy = agent.collect_policy

# a random policy to generate the first experiences that will be stored in the replay buffer
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

# %% -------Replay buffer-------

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                               batch_size=tf_train_env.batch_size,
                                                               max_length=replay_buffer_max_length)


# %% -------Start time-------
# record current timestamp
start = datetime.now()

# %% -------Initial fill-------
print("\nData collection ... \n")
collect_data(tf_train_env, random_policy, replay_buffer, initial_collect_steps, tuned_initial_exp)
dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).shuffle(
    buffer_size=replay_buffer_max_length).prefetch(3)

# smart iterator to train on all the experiences
iterator = iter(dataset)

# %% -------Training-------
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
print("\nPOLICY EVALUATION B| TRAINING \n")
avg_return = compute_avg_return(tf_test_env, agent.policy, num_test_episodes)
print("Average return (before training): ", avg_return)

returns = [avg_return]
train_rewards = []
data_save = []
average_train_rewards = []
train_loss_plot_critic = []
train_loss_plot = []
train_loss_plot_actor = []
actions_evol = []

# saver = PolicySaver(eval_policy)

# Variable pour suivre la meilleure performance
best_avg_return = avg_return



# %%
i = 0

print("\nTRAINING \n")
for _ in range(num_training_episodes):

    for _ in range(nb_quarters_per_episode):

        # Collect a few steps using collect_policy and save to the replay buffer. ! Tuned = 1 is compulsory for this line !
        rew, act = collect_data(tf_train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration, 1)

        train_rewards.append(rew)

        # Sample a batch of data from the buffer and update the agent's networks.
        experience, unused_info = next(iterator)

        train_loss = agent.train(experience).loss

        train_loss_plot.append(train_loss)

        step = agent.train_step_counter.numpy()

        i += 1

        if step % log_interval == 0:
            average_train_rewards.append(np.sum(train_rewards) / (log_interval * collect_steps_per_iteration))
            print('step = {0}: loss = {1}\taverage reward = {2}'.format(step, int(train_loss),
                                                                        int(average_train_rewards[-1])))
            train_rewards = []

        if step % eval_interval == 0:
            avg_return = compute_avg_return(tf_test_env, agent.policy, num_test_episodes)
            print('step = {0}: Average Return val = {1}'.format(step, int(avg_return)))
            returns.append(avg_return)

            if avg_return > best_avg_return:
                best_avg_return = avg_return
                print(f'New best policy with return: {best_avg_return}')


# %% -------End of training time-------
end = datetime.now()
td = (end - start).total_seconds()
print(f"The time of execution is : {td:.03f}s")


# %% -------plot + save fig-------
plotdata(average_train_rewards, log_interval, "Training")
plotdata(returns, eval_interval, "test")
plt.plot(train_loss_plot)
