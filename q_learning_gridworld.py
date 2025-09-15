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


class Environment:

    def __init__(self):
        """
        Initializes the environment parameters.
        
        Args:
        
        Returns:
        """
        pass
    
    def next_state(self):
        """
        Computes the next state given the currect state and action.
        
        Args:
        
        Returns:
        """
        pass

    def valid_actions(self):
        """
        Specifies which actions the agent can choose in a given state.
        
        Args:
        
        Returns:
        """
        pass

    def reset(self):
        """
        Resets the environment to the initial state.
        
        Args:
        
        Returns:
        """
        pass

    def step(self):
        """
        Applies the agent's actions and updates the environment.
        
        Args:
        
        Returns:
        """
        pass

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
    def __init__(self):
        """
        Initializes agent parameters and Q-Table.
        
        Args:
        
        Returns:
        """
        pass

    def epsilon_greedy(self):
        """
        Implements epsilon-greedy policy to dictate the agent's actions.

        Args:

        Returns:
        """
        pass
        
    def update_q_value(self):
        """
        Updates Q-table after agent actions using the Q-function update rule.
        
        Args:
        
        Returns:
        """
        pass


def train_agent():
    """
    Training loop where the agent updates the Q-Table.
    
    Args:
    
    Returns:
    """
    pass


def evaluate_agent():
    """
    Evaluation loop (no learning occurs).
    
    Args:
    
    Returns:
    """
    pass


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
        "num_episodes": 1000,  # Number of training episodes
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

    # Defining Constants - PyGame Rendering
    rendering_parameters = {
        "screen_width": 500,
        "screen_height": 500
    }

    # Creating a wrapped dictionary to return
    config_dict = {
        "training": training_parameters,
        "states": state_conditions,
        "rendering": rendering_parameters
    }

    return config_dict


def main():
    """
    Loads configurations, creates the environment and agent, and calls to either train or evaluate.
    
    Args:
    
    Returns:
    """
    
    config = configurations()

    pass


if __name__ == "__main__":
    main()
        