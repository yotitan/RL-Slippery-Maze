import logging
import random
from datetime import datetime

import numpy as np
import pandas as pd

from environment import Status
from models import AbstractModel


class QLearnFixedHorizon(AbstractModel):
    """ Tabular Q-learning prediction model.

        For every state (here: the agents current location ) the value for each of the actions is stored in a table.
        The key for this table is (state + action). Initially all values are 0. When playing training games
        after every move the value in the table is updated based on the reward gained after making the move. Training
        ends after a fixed number of games, or earlier if a stopping criterion is reached (here: a 100% win rate).
    """
    default_check_convergence_every = 5  # by default check for convergence every # episodes

    def __init__(self, game, **kwargs):
        """ Create a new prediction model for 'game'.

        :param class Maze game: Maze game object
        :param kwargs: model dependent init parameters
        """
        super().__init__(game, name="QTableModel", **kwargs)
        self.Q = dict()  # table with value for (state, action) combination

    def train(self, stop_at_convergence=False, **kwargs):
        """ Train the model.

            :param stop_at_convergence: stop training as soon as convergence is reached

            Hyperparameters:
            :keyword float discount: (gamma) preference for future rewards (0 = not at all, 1 = only)
            :keyword float exploration_rate: (epsilon) 0 = preference for exploring (0 = not at all, 1 = only)
            :keyword float exploration_decay: exploration rate reduction after each random step (<= 1, 1 = no at all)
            :keyword float learning_rate: (alpha) preference for using new knowledge (0 = not at all, 1 = only)
            :keyword int episodes: number of training games to play
            :return int, datetime: number of training episodes, total time spent
        """
        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        exploration_decay = kwargs.get("exploration_decay", 0.995)  # % reduction per step = 100 - exploration decay
        learning_rate = kwargs.get("learning_rate", 0.10)
        episodes = max(kwargs.get("episodes", 1000), 1)
        horizon = max(kwargs.get("horizon", 1), 1)
        check_convergence_every = kwargs.get("check_convergence_every", self.default_check_convergence_every)

        # variables for reporting purposes
        cumulative_reward = 0
        cumulative_reward_history = []
        win_history = []

        start_list = list()
        start_time = datetime.now()

        mean_episode_length = 0

        maze_rows, maze_cols = np.shape(self.environment.maze)

        num_actions = len(self.environment.actions)
        fhtd_q_table = np.zeros((maze_rows, maze_cols, len(self.environment.actions), horizon + 1))
        fhtd_q_table[:, :, :, 0] = 0.

        # training starts here
        for episode in range(1, episodes + 1):
            # optimization: make sure to start from all possible cells
            if not start_list:
                start_list = self.environment.empty.copy()
            start_cell = random.choice(start_list)
            start_list.remove(start_cell)

            state = self.environment.reset(start_cell)
            state = tuple(state.flatten())  # change np.ndarray to tuple so it can be used as dictionary key

            while True:
                mean_episode_length += 1

                # choose action epsilon greedy (off-policy, instead of only using the learned policy)
                if np.random.random() < exploration_rate:
                    action = random.choice(self.environment.actions)
                else:
                    action = self.predict(state)

                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())

                cumulative_reward += reward
                
                ######################################################################
                for h in range(1, horizon + 1):

                    # Estimate V_(h-1)(s_(t+1)): the reward accrued in h-1 steps
                    # from the next state
                    if status == Status.PLAYING:
                        future_q = np.max([fhtd_q_table[next_state[0], next_state[1], action_i, h - 1] for action_i in range(num_actions)])
                    else:
                        future_q = 0.

                    # Bootstrap the V_h value off the future value estimate for the
                    # next state, added to the reward received in getting to that
                    # next state
                    q_target = reward + discount * future_q

                    # Update the V_h estimate towards the future estimate
                    fhtd_q_table[state[0], state[1], action, h] += learning_rate * (q_target - fhtd_q_table[state[0], state[1], action, h])
                ######################################################################

                if (state, action) not in self.Q.keys():  # ensure value exists for (state, action) to avoid a KeyError
                    self.Q[(state, action)] = 0.0

                # max_next_Q = max([self.Q.get((next_state, a), 0.0) for a in self.environment.actions])
                # self.Q[(update_state, update_action)] += learning_rate * (reward_term + discount ** horizon * max_next_Q - self.Q[(update_state, update_action)])

                for idx, val in np.ndenumerate( fhtd_q_table[:,:,:,-1] ):
                    # print( idx[0:2], idx[2], val )
                    self.Q[(idx[0:2], idx[2])] = val

                if status in (Status.WIN, Status.LOSE):  # terminal state reached, stop training episode
                    break

                state = next_state

                self.environment.render_q(self)

            cumulative_reward_history.append(cumulative_reward)

            logging.info("episode: {:d}/{:d} | status: {:4s} | e: {:.5f}"
                         .format(episode, episodes, status.name, exploration_rate))

            if episode % check_convergence_every == 0:
                # check if the current model does win from all starting cells
                # only possible if there is a finite number of starting states
                w_all, win_rate = self.environment.check_win_all(self)
                win_history.append((episode, win_rate))
                if w_all is True and stop_at_convergence is True:
                    logging.info("won from all start cells, stop learning")
                    break

            exploration_rate *= exploration_decay  # explore less as training progresses

        logging.info("episodes: {:d} | time spent: {}".format(episode, datetime.now() - start_time))

        mean_episode_length /= episode
        logging.info('cumulative reward: {:f} | mean episode length: {:f}'.format(cumulative_reward, mean_episode_length))

        if stop_at_convergence is False:
            model = 'QLEARN'
            slippery = self.environment.slippery

            if learning_rate == 0.1:
                # Record Cumulative Reward History
                cumulative_reward_df = pd.DataFrame()
                cumulative_reward_df['Episode'] = np.arange(1, episode + 1)
                cumulative_reward_df['CumulativeReward'] = cumulative_reward_history
                cumulative_reward_df['Horizon'] = horizon
                cumulative_reward_df['Slippery'] = slippery
                cumulative_reward_df['Model'] = model

                cumulative_reward_file = pd.read_excel('CumulativeReward.xlsx')
                cumulative_reward_file = cumulative_reward_file[
                                                                    (cumulative_reward_file['Horizon']!=horizon) |
                                                                    (cumulative_reward_file['Slippery']!=slippery) |
                                                                    (cumulative_reward_file['Model']!=model)
                                                                ].reset_index(drop=True)

                cumulative_reward_file = pd.concat([cumulative_reward_file, cumulative_reward_df], ignore_index=True)
                cumulative_reward_file.to_excel('CumulativeReward.xlsx', index=False)

                logging.info('Updated Cumulative Reward History')

                # Record Win Rate History
                win_history_df = pd.DataFrame(win_history, columns=['Episode', 'WinRate'])
                win_history_df['Horizon'] = horizon
                win_history_df['Slippery'] = slippery
                win_history_df['Model'] = model

                win_history_file = pd.read_excel('WinRate.xlsx')
                win_history_file = win_history_file[
                                                        (win_history_file['Horizon']!=horizon) |
                                                        (win_history_file['Slippery']!=slippery) |
                                                        (win_history_file['Model']!=model)
                                                    ].reset_index(drop=True)

                win_history_file = pd.concat([win_history_file, win_history_df], ignore_index=True)
                win_history_file.to_excel('WinRate.xlsx', index=False)

                logging.info('Updated Win Rate History')

            # Record Mean Episode Length
            mean_episode_length_df = pd.DataFrame()
            mean_episode_length_df['Slippery'] = [slippery]
            mean_episode_length_df['Model'] = model
            mean_episode_length_df['Horizon'] = horizon
            mean_episode_length_df['Alpha'] = learning_rate
            mean_episode_length_df['MeanEpisodeLength'] = mean_episode_length

            mean_episode_length_file = pd.read_excel('MeanEpisodeLength.xlsx')
            mean_episode_length_file = mean_episode_length_file[
                                                    (mean_episode_length_file['Horizon']!=horizon) |
                                                    (mean_episode_length_file['Slippery']!=slippery) |
                                                    (mean_episode_length_file['Model']!=model) |
                                                    (mean_episode_length_file['Alpha']!=learning_rate)
                                                ].reset_index(drop=True)

            mean_episode_length_file = pd.concat([mean_episode_length_file, mean_episode_length_df], ignore_index=True)
            mean_episode_length_file.to_excel('MeanEpisodeLength.xlsx', index=False)

            logging.info('Updated Mean Episode Length History')

        return cumulative_reward_history, win_history, episode, datetime.now() - start_time

    def q(self, state):
        """ Get q values for all actions for a certain state. """
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        return np.array([self.Q.get((state, action), 0.0) for action in self.environment.actions])

    def predict(self, state):
        """ Policy: choose the action with the highest value from the Q-table.
            Random choice if multiple actions have the same (max) value.

            :param np.ndarray state: game state
            :return int: selected action
        """
        q = self.q(state)

        logging.debug("q[] = {}".format(q))

        actions = np.nonzero(q == np.max(q))[0]  # get index of the action(s) with the max value
        return random.choice(actions)
