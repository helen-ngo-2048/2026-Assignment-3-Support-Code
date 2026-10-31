import sys
import time
import random
import itertools
from collections import deque

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
    QL_LEARNING_RATE = 0.2

    FQ_EPSILON_START = 0.2
    FQ_EPSILON_END = 0.1
    FQ_LEARNING_RATE = 0.1

    # assume a typical boost of about 2 tiles
    BOOST_LANDING_DISTANCE = 2

    # (row change, col change) of each action
    ACTION_DELTAS = {
        GameEnv.WALK_LEFT: (0, -1), GameEnv.BOOST_LEFT: (0, -1), GameEnv.JUMP_LEFT: (0, -1),
        GameEnv.WALK_RIGHT: (0, 1), GameEnv.BOOST_RIGHT: (0, 1), GameEnv.JUMP_RIGHT: (0, 1),
        GameEnv.WALK_UP: (-1, 0), GameEnv.BOOST_UP: (-1, 0), GameEnv.JUMP_UP: (-1, 0),
        GameEnv.WALK_DOWN: (1, 0), GameEnv.BOOST_DOWN: (1, 0), GameEnv.JUMP_DOWN: (1, 0),
    }

    def __init__(self, game_env: GameEnv):
        self.game_env = game_env
        self.q_values = {}
        self.epsilon = None
        self.ql_episode_rewards = []
        self.fq_weights = []
        self.fq_episode_rewards = []
        self.target_distances = {}
        self.distance_to_go = {}

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

        # total reward of each finished training episode (for report's learning curves)
        self.ql_episode_rewards = []

    def ql_train_episode(self):
        """
        Run one episode of tabular Q-learning.
        """
        state = self.game_env.get_init_state()
        episode_reward = 0.0
        
        for step in range(self.game_env.max_episode_steps):
            # choose an action exploration/exploitation and perform
            action = self.ql_epsilon_greedy_action(state)
            next_state, reward, err_msg = self.game_env.perform_action(state, action)
            episode_reward += reward

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

        self.ql_episode_rewards.append(episode_reward)
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
        self.fq_weights = [0.0, 0.0]
        self.epsilon = self.FQ_EPSILON_START
        self.fq_episode_rewards = []

        # BFS distances from every crystal and launch site to every tile
        self.target_distances = {}
        for target in self.game_env.crystal_positions + self.game_env.launch_positions:
            self.target_distances[target] = self.bfs_distances(target)

        # greedy route distance of every state to avoid search during training
        self.distance_to_go = {}
        for state in self.get_all_states():
            self.distance_to_go[state] = self.get_greedy_route_distance(state)

    def fq_get_features(self, state: GameState, action):
        """
        Return the feature vector of the given (state, action) pair, so that Q(s, a) = w . features(s, a).
        :param state: the current state
        :param action: an element of GameEnv.ACTIONS
        :return: a list (or 1-D numpy array) of at most 256 numbers, the same length for every (state, action) pair
        """
        landing_state = self.get_landing_state(state, action)
        return [1.0, -self.distance_to_go[landing_state]]

    def fq_train_episode(self):
        """
        Run one episode of feature-based Q-learning (Q-learning with linear function approximation).
        """
        state = self.game_env.get_init_state()
        episode_reward = 0.0

        for step in range(self.game_env.max_episode_steps):
            # choose an action exploration/exploitation and perform
            action = self.fq_epsilon_greedy_action(state)
            next_state, reward, err_msg = self.game_env.perform_action(state, action)
            episode_reward += reward

            terminal = self.game_env.is_terminal(next_state)
            self.update_weights(state, action, reward, next_state, terminal)

            state = next_state
            if terminal:
                break

        self.fq_episode_rewards.append(episode_reward)
        self.fq_epsilon_decay()

    def fq_select_action(self, state: GameState):
        """
        Select the greedy action for the given state according to your learned weights.
        :param state: the current state
        :return: greedy action for the given state (element of GameEnv.ACTIONS)
        """
        # find the highest Q-value among this state's valid actions, breaking ties randomly
        best_q = float('-inf')
        best_actions = []
        for action in self.game_env.get_valid_actions(state):
            q_value = self.get_q_value(state, action)
            if q_value > best_q:
                best_q = q_value
                best_actions = [action]
            elif q_value == best_q:
                best_actions.append(action)
        return random.choice(best_actions)

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

    def bfs_distances(self, target):
        """
        BFS out from target, going around Rock and Lava tiles.
        :param target: (row, col) of the tile to measure distances to
        :return: dict mapping (row, col) to the number of steps to target (leave out unreachable tiles)
        """
        distances = {target: 0}
        queue = deque([target])
        while len(queue) > 0:
            row, col = queue.popleft()
            for delta_row, delta_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbour = (row + delta_row, col + delta_col)
                if neighbour in distances:
                    continue
                if not (0 <= neighbour[0] < self.game_env.n_rows and 0 <= neighbour[1] < self.game_env.n_cols):
                    continue
                tile = self.game_env.grid_data[neighbour[0]][neighbour[1]]
                if tile == self.game_env.ROCK_TILE or tile == self.game_env.LAVA_TILE:
                    continue
                distances[neighbour] = distances[(row, col)] + 1
                queue.append(neighbour)
        return distances

    def get_tile_distance(self, position, target):
        """
        Steps from position to target, using the precomputed BFS distances. Tiles with no route get 
        distance > any real route.
        """
        unreachable = self.game_env.n_rows * self.game_env.n_cols
        return self.target_distances[target].get(position, unreachable)

    def get_nearest_target(self, position, targets):
        """
        :return: (nearest target to position, distance to it)
        """
        nearest = None
        nearest_distance = float('inf')
        for target in targets:
            distance = self.get_tile_distance(position, target)
            if distance < nearest_distance:
                nearest = target
                nearest_distance = distance
        return nearest, nearest_distance

    def get_greedy_route_distance(self, state):
        """
        A3 Helper Guide: keep walking to the nearest uncollected crystal until there are enough
        crystals, then walk to the nearest launch site.
        :return: total steps along that route
        """
        position = (state.row, state.col)
        remaining_crystals = []
        for i in range(self.game_env.n_crystals):
            if state.crystal_status[i] == 0:
                remaining_crystals.append(self.game_env.crystal_positions[i])

        total = 0
        crystals_needed = self.game_env.min_samples - sum(state.crystal_status)
        for _ in range(crystals_needed):
            crystal, distance = self.get_nearest_target(position, remaining_crystals)
            total += distance
            position = crystal
            remaining_crystals.remove(crystal)

        launch_site, distance = self.get_nearest_target(position, self.game_env.launch_positions)
        return total + distance

    def get_landing_state(self, state, action):
        """
        The state the intended move ends in, ignoring drift and double moves. Boosts are assumed to travel
        BOOST_LANDING_DISTANCE tiles. The move stops before Rock and on a Crater or Lava tile.
        """
        delta_row, delta_col = self.ACTION_DELTAS[action]
        distance = 1
        if action in self.game_env.BOOST_ACTIONS:
            distance = self.BOOST_LANDING_DISTANCE

        row, col = state.row, state.col
        for _ in range(distance):
            next_row = row + delta_row
            next_col = col + delta_col
            if not (0 <= next_row < self.game_env.n_rows and 0 <= next_col < self.game_env.n_cols):
                break
            if self.game_env.grid_data[next_row][next_col] == self.game_env.ROCK_TILE:
                break
            row = next_row
            col = next_col
            tile = self.game_env.grid_data[row][col]
            if tile == self.game_env.CRATER_TILE or tile == self.game_env.LAVA_TILE:
                break

        # landing on an uncollected crystal collects it
        crystal_status = state.crystal_status
        if (row, col) in self.game_env.crystal_positions:
            crystal_status = list(crystal_status)
            crystal_status[self.game_env.crystal_positions.index((row, col))] = 1
            crystal_status = tuple(crystal_status)
        return GameState(row, col, crystal_status)

    def get_q_value(self, state, action):
        """
        Return the Q-value of the given (state, action) pair.
        """
        features = self.fq_get_features(state, action)
        q_value = 0.0
        for i in range(len(features)):
            q_value += self.fq_weights[i] * features[i]
        return q_value

    def update_weights(self, state, action, reward, next_state, terminal):
        """
        Update the feature weights from one step of experience.
        """
        target = reward
        if not terminal:
            target += self.game_env.gamma * self.get_q_value(next_state, self.fq_select_action(next_state))
        td_error = target - self.get_q_value(state, action)

        features = self.fq_get_features(state, action)
        squared_size = 0.0
        for feature in features:
            squared_size += feature * feature

        step = self.FQ_LEARNING_RATE * td_error / squared_size
        for i in range(len(features)):
            self.fq_weights[i] += step * features[i]

    def fq_epsilon_greedy_action(self, state):
        """
        Use epsilon-greedy to select an action for a state during training.
        With probability epsilon explore (random valid action), otherwise exploit (greedy action).
        """
        if random.random() < self.epsilon:
            return random.choice(self.game_env.get_valid_actions(state))
        return self.fq_select_action(state)

    def fq_epsilon_decay(self):
        """
        Decay epsilon linearly over the training step budget.
        """
        progress = self.game_env.training_steps_used / self.game_env.training_step_budget
        self.epsilon = max(self.FQ_EPSILON_END, self.FQ_EPSILON_START * (1 - progress))
