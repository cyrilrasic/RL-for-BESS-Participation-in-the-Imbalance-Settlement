# -*- coding: utf-8 -*-
"""
TD3 classes & functions - reprise TFE pour papier conf

Created on Wed Nov  6 14:01:28 2024

@author: Cyril
"""
# %% -------import-------
# from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
from numpy.matlib import zeros
import numpy as np
import tensorflow as tf
import tf_agents.trajectories.policy_step as ps
from tf_agents.trajectories import trajectory
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from math import floor, ceil
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class Environment(py_environment.PyEnvironment):
    def __init__(
        self, observations, non_observable_states, scaler, scalers,
        max_power: float, discount_rate: float, nb_quarters_per_episode: int,
        col_MDP: int, EP: float, eta: float, bat_replacement_cost, penalty
    ):
        super().__init__()

        # Specs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.float32, minimum=-1.0, maximum=1.0, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(observations.shape[1],), dtype=np.float32,
            minimum=-1.1, maximum=1.1, name="observation"
        )

        # Données principales
        self._observation_samples = np.asarray(observations, dtype=np.float32)
        self._non_observable_samples = np.asarray(non_observable_states, dtype=np.float32)

        # Constantes
        self._nb_quarters_per_episode = nb_quarters_per_episode
        self._scaler_si = scaler
        self._scalers_MP = scalers
        self._SOC_index = 28
        self._max_power = float(max_power)
        self._discount = float(discount_rate)
        self._col_MDP = col_MDP
        self._EP_ratio = float(EP)
        self._eta = float(eta)
        self._R = float(bat_replacement_cost)
        self._penalty = float(penalty)

        # États internes
        self._observation_index = 0
        self._episode = 0
        self._episode_ended = False
        self._observation = self._observation_samples[self._observation_index]

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode += 1
        self._episode_ended = False
        # SoC initial
        self._observation_samples[self._observation_index, self._SOC_index] = 0.5
        return ts.restart(self._observation)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        initial_soc = self._observation[self._SOC_index]
        # Nouveau SoC proposé
        new_soc = initial_soc + (action / (4 * self._EP_ratio))

        # Pénalité si hors bornes
        if 0 <= new_soc <= 1:
            penalized = 0.0
        else:
            penalized = self._penalty

        new_soc = np.clip(new_soc, 0.0, 1.0)

        # Fraction de puissance ajustée au SoC
        fraction_of_max_power = (new_soc - initial_soc) * (4 * self._EP_ratio)

        # Coût de décharge
        cost = 0.0
        if fraction_of_max_power < 0:
            cost = self._R * 5.24e-4 * (abs(fraction_of_max_power) / 4) ** 2.03

        # Puissance batterie
        battery_charge_MW = fraction_of_max_power * self._max_power

        # SI réel
        syst_imb = self._scaler_si.inverse_transform(
            self._non_observable_samples[self._observation_index].reshape(1, 1)
        )[0, 0]

        # Charge réseau
        if battery_charge_MW > 0:
            network_charge_MW = battery_charge_MW / np.sqrt(self._eta)
        else:
            network_charge_MW = battery_charge_MW * np.sqrt(self._eta)

        real_si = syst_imb - network_charge_MW

        # Index prix
        index = (-real_si / 100) + 6
        index = floor(index) if index < 6 else ceil(index)
        index = int(np.clip(index, 0, 12))

        MP_price = self._scalers_MP[index].inverse_transform(
            np.array([[self._observation[self._col_MDP + index - 1]]], dtype=np.float32)
        )[0, 0]

        # Profit et reward
        profit = MP_price * (-network_charge_MW / 4)
        reward = profit - cost - penalized

        # Avance l’index
        self._observation_index += 1

        # Cas 1 : pas à la fin du dataset
        if self._observation_index < len(self._observation_samples):
            self._observation_samples[self._observation_index, self._SOC_index] = new_soc
            self._observation = self._observation_samples[self._observation_index]

            if self._observation_index < (self._nb_quarters_per_episode * self._episode):
                return ts.transition(self._observation, reward, discount=self._discount)
            else:
                self._episode_ended = True
                return ts.termination(self._observation, reward)

        # Cas 2 : fin du dataset → restart
        self._observation_index = 0
        self._episode = 0
        self._observation = self._observation_samples[self._observation_index]
        self._episode_ended = True
        return ts.termination(self._observation, reward)




class Environmentvalidation(py_environment.PyEnvironment):
    def __init__(
        self, observations, non_observable_states, scaler, scalers,
        max_power: float, discount_rate: float, nb_quarters_per_episode: int,
        col_MDP: int, EP: float, eta: float, bat_replacement_cost
    ):
        super().__init__()

        # Specs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.float32, minimum=-1.0, maximum=1.0, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(observations.shape[1],), dtype=np.float32,
            minimum=-1.1, maximum=1.1, name="observation"
        )

        # Données (stockées en float32 pour limiter la mémoire)
        self._observation_samples = np.asarray(observations, dtype=np.float32)
        self._non_observable_samples = np.asarray(non_observable_states, dtype=np.float32)

        # Constantes
        self._nb_quarters_per_episode = nb_quarters_per_episode
        self._scaler_si = scaler
        self._scalers_MP = scalers
        self._SOC_index = 28
        self._max_power = float(max_power)
        self._discount = float(discount_rate)
        self._col_MDP = col_MDP
        self._EP_ratio = float(EP)
        self._eta = float(eta)
        self._R = float(bat_replacement_cost)

        # États internes
        self._observation_index = 0
        self._episode_ended = False
        self._episode = 0
        self._observation = self._observation_samples[self._observation_index]

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode += 1
        self._episode_ended = False

        # SoC réinitialisé
        self._observation_samples[self._observation_index, self._SOC_index] = 0.5
        return ts.restart(self._observation)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        # Action
        fraction_of_max_power = float(action)

        # Nouveau SoC
        soc = self._observation[self._SOC_index]
        new_soc = soc + (fraction_of_max_power / (4 * self._EP_ratio))
        new_soc = np.clip(new_soc, 0.0, 1.0)

        # Ajustement de l’action
        fraction_of_max_power = (new_soc - soc) * (4 * self._EP_ratio)

        # Coût si décharge
        cost = 0.0
        if action < 0:
            cost = self._R * 5.24e-4 * (abs(fraction_of_max_power) / 4) ** 2.03

        # Puissance batterie
        battery_charge_MW = fraction_of_max_power * self._max_power

        # SI réel
        syst_imb = self._scaler_si.inverse_transform(
            self._non_observable_samples[self._observation_index].reshape(1, 1)
        )[0, 0]

        # Charge réseau
        if battery_charge_MW > 0:
            network_charge_MW = battery_charge_MW / np.sqrt(self._eta)
        else:
            network_charge_MW = battery_charge_MW * np.sqrt(self._eta)

        real_si = syst_imb - network_charge_MW

        # Index prix
        index = (-real_si / 100) + 6
        index = floor(index) if index < 6 else ceil(index)
        index = int(np.clip(index, 0, 12))

        MP_price = self._scalers_MP[index].inverse_transform(
            np.array([[self._observation[self._col_MDP + index - 1]]], dtype=np.float32)
        )[0, 0]

        # Profit et reward
        profit = MP_price * (-network_charge_MW / 4)
        reward = profit - cost

        # Avance l’index
        self._observation_index += 1

        # Pas encore à la fin du dataset
        if self._observation_index < len(self._observation_samples):
            self._observation_samples[self._observation_index, self._SOC_index] = new_soc
            self._observation = self._observation_samples[self._observation_index]

            if self._observation_index < (self._nb_quarters_per_episode * self._episode):
                return ts.transition(self._observation, reward, discount=self._discount)
            else:
                self._episode_ended = True
                return ts.termination(self._observation, reward)

        # Fin du dataset → restart
        self._observation_index = 0
        self._episode = 0
        self._observation = self._observation_samples[self._observation_index]
        self._episode_ended = True
        return ts.termination(self._observation, reward)


# %% -------Usefull func-------
def floor(x):
    return int(np.floor(x))


def ceil(x):
    return int(np.ceil(x))


def plotdata(y_data, interval, title):
    iterations = np.arange(len(y_data)) * interval
    plt.plot(iterations, y_data)
    plt.title(title)
    plt.ylabel('Average Return (per eps)')
    plt.xlabel('iterations')
    plt.ylim()
    plt.grid()
    plt.show()


def std_around_value(data, arbitrary_value):
    deviations = np.array(data) - arbitrary_value
    return np.sqrt(np.mean(deviations ** 2))


# %% -------Import data and preprocessing-------
def importdata(path_training, path_validation, path_test, num_future):
    LA_steps = num_future-1
    # number of the column containing the SI
    index_col_si = 0

    # number of the columns, containing the MPs
    index_first_col_MDP = 16
    index_last_col_MIP = index_first_col_MDP + 12  # +12 as 13 prices

    df = pd.read_excel(path_training, usecols="B:AD")

    # add a column of 0 for the soc
    df = df.assign(soc=np.zeros(len(df)))

    train_samples = df.to_numpy().astype("float32")

    # last element of the first row = 0.5 (first soc=0.5 => start the 1st ep with a half charged battery)
    train_samples[0, -1] = 0.5

    df2 = pd.read_excel(path_validation, usecols="B:AD")
    df2 = df2.assign(soc=np.zeros(len(df2)))
    validation_samples = df2.to_numpy().astype("float32")
    validation_samples[0, -1] = 0.5

    df3 = pd.read_excel(path_test, usecols="B:AD")
    df3 = df3.assign(soc=np.zeros(len(df3)))
    test_samples = df3.to_numpy().astype("float32")
    test_samples[0, -1] = 0.5

    nb_rows_train = len(df)
    nb_rows_validation = len(df2)
    nb_rows_test = len(df3)
    nb_total_rows = nb_rows_train + nb_rows_validation + nb_rows_test

    # vertical concatenate of the samples for scaling
    samples = np.concatenate([train_samples, validation_samples, test_samples], axis=0)

    # Adding Look Ahead (LA) data: mean and std of the SI predictions quantiles
    SI_quant_mean_and_std = zeros((nb_total_rows + LA_steps, 2), dtype='float32')


    for i in range(nb_total_rows):
        SI_quant_mean_and_std[i, 0] = np.mean(samples[i, 1:11])
        SI_quant_mean_and_std[i, 1] = np.std(samples[i, 1:11])

    for i in range(LA_steps):
        SI_quant_mean_and_std[i + nb_total_rows, 0] = SI_quant_mean_and_std[i, 0]
        SI_quant_mean_and_std[i + nb_total_rows, 1] = SI_quant_mean_and_std[i, 1]

    # Adding LA_steps * 2 columns to the samples
    LA_SI_features = np.zeros((nb_total_rows, LA_steps * 2), dtype='float32')

    for i in range(nb_total_rows):
        for j in range(LA_steps):
            LA_SI_features[i, j * 2] = SI_quant_mean_and_std[i + j + 1, 0]  # Mean
            LA_SI_features[i, j * 2 + 1] = SI_quant_mean_and_std[i + j + 1, 1]  # Std

    if LA_steps != 0:
        # Concatenate the new features to the samples
        samples = np.hstack((samples, LA_SI_features))

    # Adding Look Ahead (LA) data: MP corresponding to the mean SI prediction and std of the price around this price
    MP_mean_and_std = zeros((nb_total_rows + LA_steps, 2), dtype='float32')

    for i in range(nb_total_rows):
        index_MP = (-1 * SI_quant_mean_and_std[i, 0] / 100) + 6
        index_MP = np.clip(index_MP, 0, 12)

        if index_MP < 6:
            index_MP = floor(index_MP)
        else:
            index_MP = ceil(index_MP)

        MP_mean_and_std[i, 0] = samples[i, index_MP + index_first_col_MDP]  # mean
        MP_mean_and_std[i, 1] = std_around_value(samples[i, index_first_col_MDP: index_last_col_MIP],
                                                 MP_mean_and_std[i, 0])  # std around mean

    for i in range(LA_steps):
        MP_mean_and_std[i + nb_total_rows, 0] = MP_mean_and_std[i, 0]
        MP_mean_and_std[i + nb_total_rows, 1] = MP_mean_and_std[i, 1]

    # Adding LA_steps * 2 columns to the samples
    LA_MP_features = np.zeros((nb_total_rows, LA_steps * 2), dtype='float32')

    for i in range(nb_total_rows):
        for j in range(LA_steps):
            LA_MP_features[i, j * 2] = MP_mean_and_std[i + j + 1, 0]  # Mean
            LA_MP_features[i, j * 2 + 1] = MP_mean_and_std[i + j + 1, 1]  # Std

    if LA_steps != 0:
        # Concatenate the new features to the samples
        samples = np.hstack((samples, LA_MP_features))

    # all the SI of the excel in non_obs_exact_SI
    non_observable_exact_si = samples[:, index_col_si]

    # obs_samp = samp without SI => the SI is considered non_observable, but we have the quantiles of the predicted SI
    observable_samples = np.delete(samples, index_col_si, 1)

    scaled_observable_samples = observable_samples

    # We scale down the SI prediction data |b| 0 and 1
    scaler_si = MinMaxScaler(feature_range=(-1, 1))  # creation of the object
    # fit_transform takes only the column with the SI prediction
    scaled_si = scaler_si.fit_transform(non_observable_exact_si.reshape(-1, 1))

    scaler_list = []
    # for j in 0,...,10,15,...,(15+13) => permet de ne pas scale les variables calendaires
    # on scale toutes les variables observables et on stock les scalers des MP dans une liste pour
    # pouvoir ensuite les utiliser pour la tsfo inverse.
    # Toutes les données scaled sont stockées dans scaled_observable_samples
    for j in np.concatenate((np.arange(11), np.arange(index_first_col_MDP - 1, index_last_col_MIP),
                             np.arange(index_last_col_MIP + 1, index_last_col_MIP + (4 * LA_steps) + 1))):
        scaler_MP = MinMaxScaler(feature_range=(0, 1))
        scaled_MP = scaler_MP.fit_transform(observable_samples[:, j].reshape(-1, 1))
        scaled_observable_samples[:, j] = scaled_MP[:, 0]
        # In scaler_list we just need the scalers of the MPs for the inverse transform
        if index_first_col_MDP - 1 <= j <= index_last_col_MIP - 1:
            scaler_list.append(scaler_MP)

    # return to training, validation and test sets
    train_observable_samples = scaled_observable_samples[0:nb_rows_train, :]
    validation_observable_samples = scaled_observable_samples[nb_rows_train:nb_rows_train + nb_rows_validation, :]
    test_observable_samples = scaled_observable_samples[
                              nb_rows_train + nb_rows_validation:nb_rows_train + nb_rows_validation + nb_rows_test, :]

    train_non_observable_samples = scaled_si[0:nb_rows_train, :]
    validation_non_observable_samples = scaled_si[nb_rows_train:nb_rows_train + nb_rows_validation, :]
    test_non_observable_samples = scaled_si[
                                  nb_rows_train + nb_rows_validation:nb_rows_train + nb_rows_validation + nb_rows_test,
                                  :]


    return index_col_si, index_first_col_MDP, scaler_si, scaler_list, train_observable_samples, validation_observable_samples, \
        test_observable_samples, train_non_observable_samples, validation_non_observable_samples, test_non_observable_samples


# %% -------Metrics and evaluation-------
# /!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\
# !!!! we provide a policy, not an agent => no cvxpy corrections !!!!!!!
# /!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\
def compute_avg_return(environment, policy, nbr_episodes):
    actions = []
    total_return = 0.0
    counter = 0
    nb_infeas = 0


    for _ in range(nbr_episodes):
        # ta=time.time()
        time_step = environment.reset()
        # tb=time.time()
        # print(tb-ta)
        episode_return = 0.0

        while not time_step.is_last():
            observation = time_step.observation.numpy()
            crt_SoC = observation[0, 28]

            action_step = policy.action(time_step)


            time_step = environment.step(action_step.action)

            # actions.append(action_step.action.numpy()[0])

            observation = time_step.observation.numpy()
            new_soc = observation[0, 28]
            # print("crt_SoC", crt_SoC)
            # print("new_soc", new_soc)
            # print("action_step", action_step)

            tol = 1e-5
            raw_action = float(action_step.action[0])
            if np.isclose(crt_SoC, new_soc, atol=tol) and not np.isclose(raw_action, 0.0, atol=0.001):
                # print('act', raw_action)
                # print('crt_SoC', crt_SoC)
                # print('new_soc', new_soc)
                nb_infeas += 1



            episode_return += time_step.reward
            counter += 1
        total_return += episode_return

    avg_return = total_return / nbr_episodes

    print('avg return: ', avg_return.numpy()[0])

    # print('actions, ',actions)
    return avg_return.numpy()[0]


# %% -------Test to save actions and reward at each step-------

# /!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\
# !!!! we provide a policy, not an agent => no cvxpy corrections !!!!!!!
# /!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\

def Visualize(environment, policy, nbr_steps):
    rewards = []
    actions = []
    soc = []

    time_step = environment.reset()

    for _ in range(nbr_steps):
        action_step = policy.action(time_step)

        observation = time_step.observation
        crt_SoCs = tf.gather(observation, indices=[28], axis=1)  # index soc = 0 for RNN !!, 28 for FCNN


        actions.append(action_step.action.numpy()[0])
        time_step = environment.step(action_step.action)
        rewards.append(time_step.reward.numpy()[0])
        soc.append(time_step.observation[-1][28].numpy())

    return actions, rewards, soc


# %% -------Data collection-------
# /!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\
# !!!! we provide a policy, not an agent => no cvxpy corrections !!!!!!!
# /!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\
def collect_data(environment, policy, buffer, steps, initial):
    # t0 = time.time()
    reward = 0
    unfeas_action = []

    # ensure a proper exploration for the first x steps by choosing the actions
    x = 16 * 20

    arbitrary_actions = [-1, -1, -0.5, 1, 1, 1, 1, 1, 1, 0.1, -1, -1, -1, -1, 0.5, -0.5]    #non-physic exp
    # arbitrary_actions = [-1, -0.5, -0.5, 1, 0.5, 0.75, 0.25, 0.5, 1, -0.1, -0.7, -0.2, 0.8, -1, 0.5, -0.5]    #physic-exp

    if initial == 0:
        for _ in range(int(x / 16)):
            for i in range(16):
                # selecting action
                # sample = np.clip(np.random.normal(0, 0.3), -1, 1)

                sample = arbitrary_actions[i]

                time_step = environment.current_time_step()
                # reward += time_step.reward
                reward = time_step.reward

                action_step = tf.constant(sample, dtype='float32', shape=(1,))
                action_step = ps.PolicyStep(action_step)

                next_time_step = environment.step(sample)

                traj = trajectory.from_transition(time_step, action_step, next_time_step)
                #traj = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32) if x.dtype == tf.float64 else x, traj)    # ensuring everything in float32 (necessary for some tf-agents operations)

                # Add trajectory to the replay buffer
                buffer.add_batch(traj)
                if next_time_step.is_last():
                    environment.reset()

                # print(action_step)
                # print(reward)
                # print(time_step.observation)

        for _ in range(steps - x):

            time_step = environment.current_time_step()
            # reward += time_step.reward
            reward = time_step.reward
            action_step = policy.action(time_step)

            observation = time_step.observation
            crt_SoCs = tf.gather(observation, indices=[28], axis=1)  # index soc = 0 for RNN !!, 28 for FCNN

            # Actions are modified only in safe projection mode
            # if SafeProjectionMode:
            #     crt_SoCs = tf.clip_by_value(crt_SoCs, clip_value_min=0.0, clip_value_max=1.0)
            #     action_step = policy_step.PolicyStep(
            #         action=tf.reshape(proj_layer(crt_SoCs, action_step.action), (1,)),
            #         state=action_step.state,
            #         info=action_step.info
            #     )

            next_time_step = environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)

            #traj = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32) if x.dtype == tf.float64 else x,traj)  # ensuring everything in float32 (necessary for some tf-agents operations)

            # Add trajectory to the replay buffer
            buffer.add_batch(traj)
            if next_time_step.is_last():
                environment.reset()

    else:
        for _ in range(steps):

            time_step = environment.current_time_step()
            reward += time_step.reward
            # reward = time_step.reward
            action_step = policy.action(time_step)

            observation = time_step.observation
            # print(observation)

            crt_SoCs = tf.gather(observation, indices=[28], axis=1)  # index soc = 0 for RNN !!, 28 for FCNN
            # Actions are modified only in safe projection mode
            # if SafeProjectionMode:
            #     crt_SoCs = tf.clip_by_value(crt_SoCs, clip_value_min=0.0, clip_value_max=1.0)
            #     action_step = policy_step.PolicyStep(
            #         action=tf.reshape(proj_layer(crt_SoCs, action_step.action), (1,)),
            #         state=action_step.state,
            #         info=action_step.info
            #     )

            next_time_step = environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            #traj = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32) if x.dtype == tf.float64 else x, traj)  # ensuring everything in float32 (necessary for some tf-agents operations)

            # Add trajectory to the replay buffer
            buffer.add_batch(traj)
            # print(action_step.action)
            if next_time_step.is_last():
                environment.reset()

    # print(time.time() - t0)
    return reward, unfeas_action


