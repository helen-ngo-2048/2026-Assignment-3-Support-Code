import sys
import time
import random
import itertools

from game_env import GameEnv
from game_state import GameState
"""
solution.py

This file is a template you should use to implement your solution.

You should implement each of the method stubs below. You may add additional methods and/or classes to this file if you
wish. You may also create additional source files and import to this file if you wish.

COMP3702 Assignment 3 "CrystalRover" Support Code

Last updated 16/10/2026
"""


class Solver:
    STUDENT_NAME = "Helen Ngo" # These are copied from A2 but was not in the original file
    STUDENT_ID = "48814553"  # same as above
    GITHUB_USERNAME = "helen-ngo-2048" # same as above

    QL_EPSILON_START = 0.2
    QL_EPSILON_END = 0.1
    QL_LEARNING_RATE = 0.05

    def __init__(self, game_env: GameEnv):
        self.game_env = game_env
        self.q_values = {}
        self.epsilon = None

    @staticmethod
    def testcases_to_attempt():
        """
        Return a list of testcase numbers you want your solution to be evaluated for.
        """
        # TODO: modify below if desired (e.g. disable larger testcases if you're having problems with RAM usage, etc)
        return [1, 2, 3, 4, 5]

    # === Training rules (apply to both algorithms) ====================================================================
    #
    # Before training starts, the tester and autograder give self.game_env a training budget: a maximum number of
    # perform_action calls, and a lowest cumulative training reward. Once either runs out, perform_action raises
    # TrainingBudgetExhausted, which ends training - do not catch it. self.game_env.training_steps_used,
    # self.game_env.training_step_budget and self.game_env.training_reward_total let you plan around the budget.
    #
    # During training, perform_action must be called on either the initial state (self.game_env.get_init_state(), which
    # starts a new episode) or the state returned by your previous perform_action call.
    #
    # Your agent must learn from experience. The movement probabilities, action costs and penalties are hidden: each
    # testcase draws its own from its seed, and the environment keeps them private. Your agent may use the map (e.g.
    # distances, or what lies in an action's direction) and the rewards it receives, but must not read the
    # environment's private parameters or call its private methods, or plan with a transition model (e.g. value
    # iteration).
    #
    # Only some actions are valid on each tile (self.game_env.get_valid_actions(state)). An invalid action leaves the
    # state unchanged and still costs its action cost.

    # === Tabular Q-learning ===========================================================================================

    def ql_initialise(self):
        """
        Initialise any variables required before the start of tabular Q-learning.
        """
        # Q-table: {state: {action: q_value}}, every Q-value starts at 0. Only the actions that are valid on the
        # state's tile get an entry. Terminal states stay in the table but never updates so their Q-values stay 0.
        self.q_values = {}
        for state in self.get_all_states():
            self.q_values[state] = {}
            for action in self.game_env.get_valid_actions(state):
                self.q_values[state][action] = 0.0

        self.epsilon = self.QL_EPSILON_START

    def ql_train_episode(self):
        """
        Run one episode of tabular Q-learning.
        """
        state = self.game_env.get_init_state()
        for step in range(self.game_env.max_episode_steps):
            # choose an action exploration/exploitation and perform
            action = self.ql_epsilon_greedy_action(state)
            next_state, reward, err_msg = self.game_env.perform_action(state, action)

            # target = r if S' is terminal, else r + gamma * max Q(S', a')
            terminal = self.game_env.is_terminal(next_state)
            if terminal:
                target = reward
            else:
                next_action = self.ql_select_action(next_state)
                target = reward + self.game_env.gamma * self.q_values[next_state][next_action]

            # Q(S, a) = Q(S, a) + learning_rate * (target - Q(S, a))
            old_q = self.q_values[state][action]
            self.q_values[state][action] = old_q + self.QL_LEARNING_RATE * (target - old_q)

            state = next_state
            if terminal:
                break

        self.ql_epsilon_decay()

    def ql_select_action(self, state: GameState):
        """
        Select the greedy action for the given state according to your learned Q-values.
        :param state: the current state
        :return: greedy action for the given state (element of GameEnv.ACTIONS)
        """
        # find the highest Q-value among this state's valid actions
        best_q = float('-inf')
        for action in self.q_values[state]:
            if self.q_values[state][action] > best_q:
                best_q = self.q_values[state][action]

        # collect every action with that Q-value, then break ties randomly
        best_actions = []
        for action in self.q_values[state]:
            if self.q_values[state][action] == best_q:
                best_actions.append(action)
        return random.choice(best_actions)

    # === Feature-based Q-learning =====================================================================================

    def fq_initialise(self):
        """
        Initialise any variables required before the start of feature-based Q-learning.
        """
        #
        # TODO: Implement any initialisation for feature-based Q-learning (e.g. initial weights, or pre-computing
        #  information about the map that your features use) here. This method is not timed, so it is the place for
        #  precomputation, but it must not call perform_action (training belongs in fq_train_episode) and must finish
        #  within 30 seconds, or the phase scores zero.
        #
        # Your weights must start all 0 or all 1 (e.g. [0.0] * number of features) and be learned during training:
        #  starting from hand-picked weights scores zero for the phase.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def fq_get_features(self, state: GameState, action):
        """
        Return the feature vector of the given (state, action) pair, so that Q(s, a) = w . features(s, a).
        :param state: the current state
        :param action: an element of GameEnv.ACTIONS
        :return: a list (or 1-D numpy array) of at most 256 numbers, the same length for every (state, action) pair
        """
        #
        # TODO: Implement your feature design here. Features that describe the situation (e.g. distances, risks) rather
        #  than identify the exact state are what let your agent act well in states it never visited during training.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def fq_train_episode(self):
        """
        Run one episode of feature-based Q-learning (Q-learning with linear function approximation).
        """
        #
        # TODO: Implement a single training episode here, as for ql_train_episode, but updating the weights of your
        #  linear Q-function instead of a Q-table.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    def fq_select_action(self, state: GameState):
        """
        Select the greedy action for the given state according to your learned weights.
        :param state: the current state
        :return: greedy action for the given state (element of GameEnv.ACTIONS)
        """
        #
        # TODO: Implement code to return the greedy action for the given state (based on your learned weights) here.
        #  This method is also evaluated from start positions your agent never trained from.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        pass

    # === Helper Methods ===============================================================================================

    def ql_epsilon_greedy_action(self, state):
        """
        Use epsilon-greedy to select an action for a state during training.
        With probability epsilon explore (random valid action), otherwise exploit (greedy action).
        """
        if random.random() < self.epsilon:
            valid_actions = list(self.q_values[state].keys())
            return random.choice(valid_actions)
        return self.ql_select_action(state)

    def ql_epsilon_decay(self):
        """
        Decay epsilon linearly over the training step budget, from QL_EPSILON_START down to QL_EPSILON_END.
        """
        progress = self.game_env.training_steps_used / self.game_env.training_step_budget
        self.epsilon = max(self.QL_EPSILON_END, self.QL_EPSILON_START * (1 - progress))

    def get_all_states(self):
        """
        Build the list of every state: each non-Rock tile combined with every possible crystal_status tuple.
        Return list of GameState
        """
        # every way the crystals can be collected. 2 crystals -> (0, 0), (0, 1), (1, 0), (1, 1)
        crystal_statuses = itertools.product([0, 1], repeat=self.game_env.n_crystals)
        crystal_statuses = list(crystal_statuses)

        states = []
        for row in range(self.game_env.n_rows):
            for col in range(self.game_env.n_cols):
                # rover can never stand on a rock tile
                if self.game_env.grid_data[row][col] == self.game_env.ROCK_TILE:
                    continue
                for status in crystal_statuses:
                    states.append(GameState(row, col, status))
        return states
