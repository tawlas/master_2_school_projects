import gym
import gym.envs.toy_text.frozen_lake as fl
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)


# Create environment
env = gym.make('FrozenLake-v0', is_slippery=True)

# It should print a rendering of the FrozenLake environment
env.render()

# Of course, you should RTFM (please do a quick peek into the doc)
help(fl)
# ... and again
help(gym.envs.toy_text.discrete)

# Some printing to understand the 'env' class
print("Action Space - ", env.action_space, env.nA)
print("State Space - ", env.observation_space, env.nS)
# isd is the initial state distribution
print("Initial State Distribution  - ", env.isd, np.sum(env.isd) == 1 )
print("Transitions from STATE 1 when going LEFT\n", env.P[1][0])

# If you understand those two functions
# you understand how states are linked to env.render()
def to_s(row,col):
    return row*env.ncol + col

def to_row_col(s):
    col = s % env.ncol
    row = s // env.ncol
    return row, col


# Documentation for interacting with a gym environment
help(gym.Env)





##################################################
##### STOP HERE FOR QUESTION 2 OF EXERCISE 5 #####
##################################################

# First interaction with the environment
NBR_EPISODES = 100000
HORIZON = 200
GAMMA = 0.9

VALUE_START = np.zeros(NBR_EPISODES)
for i in range(NBR_EPISODES):
    env.reset()
    done = False
    t = 0
    discount = 1
    while (not done) and (t < HORIZON):
        next_state, r, done, _ = env.step(fl.RIGHT)
        VALUE_START[i] += discount * r
        discount *= GAMMA
        t += 1

print(f"Value estimate of the starting point: {np.mean(VALUE_START)}")

offset = 10
plt.figure()
plt.title("Convergence of the Monte Carlo estimation\nof the value of the \
starting point")
plt.plot((np.cumsum(VALUE_START)/(np.arange(NBR_EPISODES)+1))[offset:])
plt.show()
