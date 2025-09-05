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