"""
The code within this script is a simple 2x2 Grid-World Implementation using the Q-learning algorithm
(the most basic RL algorithm). This implementation is exploritory in nature and will include variations
of the same problem to investigate the following basic RL concepts:
    1) Deterministic vs Stochastic Transitions
    2) Convergence Behavior
    3) Exploration Stategies
    4) Tabular vs Function Approximators
    5) Impacts of Reward Shaping

Background of Problem: You live in a 4-room house and just bought a roomba to deal with some of the 
daily dirt that has been collecting while you are busy working during the week. The roomba is only 
able to perform 5 actions: move - up, down, right, left, and clean. Each time the roomba successfully 
cleans a dirty room it gets rewarded +10, however, if it cleans a clean room it is penalized -1. 
Moving up, down, right, or left into another room results in no reward or penalization. 

Givens:
    States (S): A, B, C, D (4 total states - one for each room)
    Actions (A): up, down, right, left, clean
    Rewards (R): clean dirty = +10, clean clean = -1, move = 0
    Learning Rate (α): 0.5
    Discount Factor (γ): 0.9
"""

# Imported Packages
import copy
import random
import numpy as np
import pandas as pd
import warnings
from frozendict import frozendict

class Environment:

    def __init__(self, config):
        """
        Initializes the environment parameters.
        
        Args:
            config (dict): Contains:
                - training_parameters (dict): Contains agent parameters for learning
                - state_conditions (dict): Contains state positions and condition (dirty/clean)
                - rendering_parameters (dict): Contains constants for PyGame rendering
                
        Returns: 
            None
        """
        
        self.learning_rate = config["training"]["learning_rate"]
        self.discount_factor = config["training"]["discount_factor"]

        self.initial_state_conditions = frozendict(config["states"])  # This dict does not change (frozendict = immutable)
        self.state_conditions = dict(self.initial_state_conditions)  # This dict will change so we need a copy

        self.start_state = (0, 0)
        self.current_state = self.start_state  # initializing the current state = start state
        
    def valid_actions(self, state):
        """
        Specifies which actions the agent can choose in a given state.
        
        Args:
            state (tuple): A 2D tuple (x, y) to indicate what the current state is
        
        Returns: 
            valid_actions (list): Contains the valid actions that can be taken in the current state
        """

        x, y = state
        valid_actions = []  # Initializing a list to store the valid actions for the specific state

        if x == 0:
            valid_actions.append("right")
        if x == 1:
            valid_actions.append("left")
        if y == 0: 
            valid_actions.append("up")
        if y == 1:
            valid_actions.append("down")

        valid_actions.append("clean")  # This is always valid

        return valid_actions

    def next_state(self, state, action):
        """
        Computes the next state given the current state and action.
        
        Args:
            state (tuple): A 2D tuple (x, y) to indicate what the current state is
            action (str): Action chosen by the agent
            
        Returns:
            next_state (tuple): A 2D tuple (x, y) for the next state given action 
        """

        x, y = state

        # If action is within valid_action list then update state tuple
        if action in self.valid_actions(state):
            if action == "up":
                y = y + 1
            elif action == "down":
                y = y - 1
            elif action == "right":
                x = x + 1
            elif action == "left":
                x = x - 1
            # For cleaning action state is unchanged
            
        else:  # If action is not in valid_action list - display error
            raise ValueError("Action invalid in state: {state}")  # This will likely be unneccessary after writing agent class

        return (x, y)

    def step(self, action):
        """
        Applies the agent's actions and updates the environment.
        
        Args:
            action (str): Action chosen by the agent
        
        Returns:
            tuple:
                next_state (tuple): A 2D tuple (x, y) for the next state given action 
                reward (int): The benefit/penalty for taking the action that drives agent learning
                done (bool): A boolean to indicate if the episode is done (if true - reset) 
        """

        # Defining rewards stucture (cleaning dirty = +10, cleaning clean = -5, moving = 0)
        if action == "clean":
            if self.state_conditions[self.current_state] == "dirty":
                self.state_conditions[self.current_state] = "clean"  # Updating the current state's condition
                reward = 10
            else:  # Cell is already clean
                reward = -5
        else:  # Moving around
            reward = 0

        # Getting the next state by calling to the next_state helper function
        next_state = self.next_state(self.current_state, action)
        self.current_state = next_state  # Updating the current state

        done = False
        # Checking for terminal state (all state_conditions == "clean")
        if all(values == "clean" for values in self.state_conditions.values()):
            done = True

        return next_state, reward, done

    def reset(self):
        """
        Resets the environment to the initial state.
        
        Args:
            None
        
        Returns:
            current_state (tuple): A 2D tuple (x, y) for the current state the agent is in
            state_conditions (dict): Contains state positions and condition (dirty/clean)
        """

        # Resetting the current state and the state conditions to initialization value for next episode
        self.current_state = self.start_state
        self.state_conditions = dict(self.initial_state_conditions)

        return self.current_state, self.state_conditions

    def drawGrid(self):
        """
        Draws the GridWorld (optional visualization).
        
        Args:
        
        Returns:
        """
        pass

    def render(self):
        """
        Renders the GridWorld using a PyGame screen.
        
        Args:
        
        Returns:
        """
        pass

class QLearningAgent:
    def __init__(self, config, actions):
        """
        Initializes agent parameters and Q-Table.
        
        Args:
            config (dict): Contains:
                - training_parameters (dict): Contains agent parameters for learning
                - state_conditions (dict): Contains state positions and condition (dirty/clean)
                - rendering_parameters (dict): Contains constants for PyGame rendering
            actions (dict): All the actions an agent can take within the environment

        Returns: 
            None
        """
        self.learning_rate = config["training"]["learning_rate"]
        self.discount_factor = config["training"]["discount_factor"]
        self.epsilon = config["training"]["epsilon"]  # Using e-greedy policy this dictates probability of exploration/exploitation
        self.states = config["state_to_index"]
        self.actions = actions

        self.q_table = np.zeros((len(self.states), len(self.actions)), dtype=int)  # This needs to be a matrix of size (# states) x (# actions)

    def epsilon_greedy(self, env, current_state):
        """
        Implements epsilon-greedy policy to dictate the agent's actions.

        Args:
            env (class object): Instance of the environment
            current_state (tuple): A 2D tuple (x, y) to indicate what the current state is

        Returns:
            chosen_action (str): action chosen for the given state by epsilon_greedy policy
        """
        state_valid_actions = env.valid_actions(current_state)  # Asking environment for the state
        valid_actions_indices = [self.actions[key] for key in state_valid_actions]
        state_index = self.states[current_state]
        
        if random.uniform(0, 1) < self.epsilon:  # This is non-greedy - Explore: random action
            chosen_action = random.choice(state_valid_actions)
            return chosen_action
        
        else:  # This is the greedy option - Exploit: choose best action
            q_values = [self.q_table[state_index, action_index] for action_index in valid_actions_indices]
            best_index = np.argmax(q_values)
            chosen_action = state_valid_actions[best_index]
            return chosen_action

    def update_q_value(self, current_state, action, reward, next_state):
        """
        Updates Q-table after agent actions using the Q-function update rule.
        
        Args:
            current_state (tuple): A 2D tuple (x, y) to indicate what the current state is
            action (str): Action chosen by the agent
            next_state (tuple): A 2D tuple (x, y) for the next state given action 
            reward (int): The benefit/penalty for taking the action that drives agent learning
        
        Returns:
            None
        """
        state_index = self.states[current_state]
        next_state_index = self.states[next_state]
        action_index = self.actions[action]

        # Bellman Optimality Equation
        self.q_table[state_index, action_index] = self.q_table[state_index, action_index] + self.learning_rate*(reward + self.discount_factor*self.q_table[next_state_index].argmax() - self.q_table[state_index, action_index])


def train_agent(env, agent, num_episodes):
    """
    Training loop where the agent updates the Q-Table.
    
    Args:
        env (class object): Environment instance
        agent (class object): Agent instance
        num_episodes (int): Total episodes used for training the agent
    
    Returns:
        training_data.csv: Contains the training data for epsiode, num_steps, episode_reward, and cumulative_reward
    """

    # Initializing a Pandas dataframe to log training data
    #all_episode_data = pd.DataFrame({'Episode': 0, 'Num Steps': 0, 'Episode Reward': 0,
    #                                 'Cumulative Reward' : 0})

    columns = ['Episode', 'Num Steps', 'Episode Reward', 'Cumulative Reward']
    all_episode_data = pd.DataFrame()

    cumulative_reward = 0  # Initializing the cumulative reward during training

    # Looping for training episodes 
    for episode in range(num_episodes):

        current_state, state_conditions = env.reset()  # Resetting the environment at the start of each episode
        done = False  # Boolean for terminal state (signifies end of episode)

        episode_reward = 0  # Initializing the episode reward during training
        num_steps = 0  # Intializing a counter for number of steps for an episode

        while not done:

            action = agent.epsilon_greedy(env, current_state)  # Choosing action
            next_state, reward, done = env.step(action)  # Taking action, observing next state, reward
            agent.update_q_value(current_state, action, reward, next_state)  # Updating Q-Value
            current_state = next_state  # Moving to the next state

            episode_reward = episode_reward + reward
            num_steps = num_steps + 1

        cumulative_reward = cumulative_reward + episode_reward

        episode_data = [episode+1, num_steps, episode_reward, cumulative_reward]
        episode_data = pd.Series(episode_data)

        # Concatenating this episodes data to the all_episode_data dataframe
        all_episode_data = pd.concat([all_episode_data, episode_data.to_frame().T], keys=columns, ignore_index=True)

    # Logging all_episode_data to csv file
    all_episode_data.columns = columns
    all_episode_data.to_csv('training_data.csv', index=False)


def evaluate_agent():
    """
    Evaluation loop (no learning occurs).
    
    Args:
    
    Returns:
    """
    print("Still working on...")


def configurations():
    """
    Returns default configuration parameters.
    
    Args: None
    
    Returns:
        config_dict (dict): Contains:
            - training_parameters (dict): Contains agent parameters for learning
            - state_conditions (dict): Contains state positions and condition (dirty/clean)
            - rendering_parameters (dict): Contains constants for PyGame rendering
    """

    # Defining Constants - RL agent
    training_parameters = {
        "learning_rate": 0.5,  # alpha
        "discount_factor": 0.9,  # gamma
        "epsilon":  0.8  # used for e-greedy policy (~1 favors exploration and ~0 favors exploitation)
    
    }

    # Initial state conditions for each cell shown in the Pygame rendering using x, y coordinates for
    # better scalability
    state_conditions = {
        (0, 1): "dirty",  # State A
        (1, 1): "clean",  # State B
        (0, 0): "dirty",  # State C
        (1, 0): "clean",  # State D
    }

    states_to_index = {
        (0, 1): 0,
        (1, 1): 1,
        (0, 0): 2,
        (1, 0): 3
    }

    actions = {
        "up": 0,
        "down": 1,
        "right": 2,
        "left": 3,
        "clean": 4
    }

    # Defining Constants - PyGame Rendering
    rendering_parameters = {
        "screen_width": 500,
        "screen_height": 500
    }

    # Creating a wrapped dictionary to return
    config_dict = {
        "training": training_parameters,
        "states": state_conditions,
        "state_to_index": states_to_index,
        "rendering": rendering_parameters
    }

    return config_dict, actions


def main():
    """
    Loads configurations, creates the environment and agent, and calls to either train or evaluate.
    
    Args:
    
    Returns:
    """

    warnings.filterwarnings("ignore")
    config, actions = configurations()
    num_episodes = 1000  # Number of training episodes
    env = Environment(config)
    agent = QLearningAgent(config, actions)

    # Calling to training loop
    train_agent(env, agent, num_episodes)


if __name__ == "__main__":
    main()
        