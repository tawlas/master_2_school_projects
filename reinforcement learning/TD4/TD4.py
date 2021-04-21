#!/usr/bin/env python
# coding: utf-8

# # TD4 - Deep Q-Network

# # Tutorial - Deep Q-Learning
#
# Deep Q-Learning uses a neural network to approximate $Q$ functions. Hence, we usually refer to this algorithm as DQN (for *deep Q network*).
#
# The parameters of the neural network are denoted by $\theta$.
# *   As input, the network takes a state $s$,
# *   As output, the network returns $Q_\theta [a | s] = Q_\theta (s,a) = Q(s, a, \theta)$, the value of each action $a$ in state $s$, according to the parameters $\theta$.
#
#
# The goal of Deep Q-Learning is to learn the parameters $\theta$ so that $Q(s, a, \theta)$ approximates well the optimal $Q$-function $Q^*(s, a) \simeq Q_{\theta^*} (s,a)$.
#
# In addition to the network with parameters $\theta$, the algorithm keeps another network with the same architecture and parameters $\theta^-$, called **target network**.
#
# The algorithm works as follows:
#
# 1.   At each time $t$, the agent is in state $s_t$ and has observed the transitions $(s_i, a_i, r_i, s_i')_{i=1}^{t-1}$, which are stored in a **replay buffer**.
#
# 2.  Choose action $a_t = \arg\max_a Q_\theta(s_t, a)$ with probability $1-\varepsilon_t$, and $a_t$=random action with probability $\varepsilon_t$.
#
# 3. Take action $a_t$, observe reward $r_t$ and next state $s_t'$.
#
# 4. Add transition $(s_t, a_t, r_t, s_t')$ to the **replay buffer**.
#
# 4.  Sample a minibatch $\mathcal{B}$ containing $B$ transitions from the replay buffer. Using this minibatch, we define the loss:
#
# $$
# L(\theta) = \sum_{(s_i, a_i, r_i, s_i') \in \mathcal{B}}
# \left[
# Q(s_i, a_i, \theta) -  y_i
# \right]^2
# $$
# where the $y_i$ are the **targets** computed with the **target network** $\theta^-$:
#
# $$
# y_i = r_i + \gamma \max_{a'} Q(s_i', a', \theta^-).
# $$
#
# 5. Update the parameters $\theta$ to minimize the loss, e.g., with gradient descent (**keeping $\theta^-$ fixed**):
# $$
# \theta \gets \theta - \eta \nabla_\theta L(\theta)
# $$
# where $\eta$ is the optimization learning rate.
#
# 6. Every $N$ transitions ($t\mod N$ = 0), update target parameters: $\theta^- \gets \theta$.
#
# 7. $t \gets t+1$. Stop if $t = T$, otherwise go to step 2.

# Imports
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
import gym
from gym.wrappers import Monitor

# from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
from IPython.display import clear_output
from pathlib import Path
import base64


print(f"python --version = {sys.version}")
print(f"torch.__version__ = {torch.__version__}")
print(f"np.__version__ = {np.__version__}")
print(f"gym.__version__ = {gym.__version__}")


# ## Torch 101
#
# >"The torch package contains data structures for multi-dimensional tensors and defines mathematical operations over these tensors. Additionally, it provides many utilities for efficient serializing of Tensors and arbitrary types, and other useful utilities.
# [...] provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions."
# [PyTorch](https://pytorch.org/docs/stable/index.html)
#

# ### Variable types

# Very similar syntax to numpy.
zero_torch = torch.zeros((3, 2))

print('zero_torch is of type {:s}'.format(str(type(zero_torch))))

# Torch -> Numpy: simply call the numpy() method.
zero_np = np.zeros((3, 2))
assert (zero_torch.numpy() == zero_np).all()

# Numpy -> Torch: simply call the corresponding function on the np.array.
zero_torch_float = torch.FloatTensor(zero_np)
print('Float:\n', zero_torch_float)
zero_torch_int = torch.LongTensor(zero_np)
print('Int:\n', zero_torch_int)
zero_torch_bool = torch.BoolTensor(zero_np)
print('Bool:\n', zero_torch_bool)

# Reshape
print('View new shape...', zero_torch.view(1, 6))
# Note that print(zero_torch.reshape(1, 6)) would work too.
# The difference is in how memory is handled (view imposes contiguity).

# Algebra
a = torch.randn((3, 2))
b = torch.randn((3, 2))
print('Algebraic operations are overloaded:\n', a, '\n+\n', b, '\n=\n', a+b )

# More generally, torch shares the syntax of many attributes and functions with Numpy.


# ### Gradient management

# torch.Tensor is a similar yet more complicated data structure than np.array.
# It is basically a static array of number but may also contain an overlay to
# handle automatic differentiation (i.e keeping track of the gradient and which
# tensors depend on which).
# To access the static array embedded in a tensor, simply call the detach() method
print(zero_torch.detach())

# When inside a function performing automatic differentiation (basically when training
# a neural network), never use detach() otherwise meta information regarding gradients
# will be lost, effectively freezing the variable and preventing backprop for it.
# However when returning the result of training, do use detach() to save memory
# (the naked tensor data uses much less memory than the full-blown tensor with gradient
# management, and is much less prone to mistake such as bad copy and memory leak).

# We will solve theta * x = y in theta for x=1 and y=2
x = torch.ones(1)
y = 2 * torch.ones(1)

# Actually by default torch does not add the gradient management overlay
# when declaring tensors like this. To force it, add requires_grad=True.
theta = torch.randn(1, requires_grad=True)

# Optimisation routine
# (Adam is a sophisticated variant of SGD, with adaptive step).
optimizer = optim.Adam(params=[theta], lr=0.1)

# Loss function
print('Initial guess:', theta.detach())

for _ in range(100):
    # By default, torch accumulates gradients in memory.
    # To obtain the desired gradient descent beahviour,
    # just clean the cached gradients using the following line:
    optimizer.zero_grad()

    # Quadratic loss (* and ** are overloaded so that torch
    # knows how to differentiate them)
    loss = (y - theta * x) ** 2

    # Apply the chain rule to automatically compute gradients
    # for all relevant tensors.
    loss.backward()

    # Run one step of optimisation routine.
    optimizer.step()

print('Final estimate:', theta.detach())


# ## Setting the environment
#
# ### 1 - Define the GLOBAL parameters

# Environment
env = gym.make("CartPole-v0")

# Discount factor
GAMMA = 0.99

# Batch size
BATCH_SIZE = 256
# Capacity of the replay buffer
BUFFER_CAPACITY = 16384 # 10000
# Update target net every ... episodes
UPDATE_TARGET_EVERY = 32 # 20

# Initial value of epsilon
EPSILON_START = 1.0
# Parameter to decrease epsilon
DECREASE_EPSILON = 200
# Minimum value of epislon
EPSILON_MIN = 0.05

# Number of training episodes
N_EPISODES = 200

# Learning rate
LEARNING_RATE = 0.1


# ### 2 - Replay buffer

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)


    def __len__(self):
        return len(self.memory)

# create instance of replay buffer
replay_buffer = ReplayBuffer(BUFFER_CAPACITY)


# ### 3 - Neural Network

class Net(nn.Module):
    """
    Basic neural net.
    """
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# ### 3.5 - Loss function and optimizer

# create network and target network
hidden_size = 128
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

q_net = Net(obs_size, hidden_size, n_actions)
target_net = Net(obs_size, hidden_size, n_actions)

# objective and optimizer
objective = nn.MSELoss()
optimizer = optim.Adam(params=q_net.parameters(), lr=LEARNING_RATE)


# #### Question 0 (to do at home, not during the live session)
#
# With your own word, explain the intuition behind DQN. Recall the main parts of the aformentionned algorithm.

# ## Implementing the DQN

def get_q(states):
    """
    Compute Q function for a list of states
    """
    with torch.no_grad():
        states_v = torch.FloatTensor([states])
        output = q_net.forward(states_v).detach().numpy()  # shape (1, len(states), n_actions)
    return output[0, :, :]  # shape (len(states), n_actions)


# #### Question 1
#
# Implement the `eval_dqn` function.

def eval_dqn(n_sim=5):
    """
    ** TO BE IMPLEMENTED **

    Monte Carlo evaluation of DQN agent.

    Repeat n_sim times:
        * Run the DQN policy until the environment reaches a terminal state (= one episode)
        * Compute the sum of rewards in this episode
        * Store the sum of rewards in the episode_rewards array.
    """
    env_copy = deepcopy(env)
    episode_rewards = np.zeros(n_sim)
    return episode_rewards


# #### Question 2
#
# Implement the `choose_action` function.

def choose_action(state, epsilon):
    """
    ** TO BE IMPLEMENTED **

    Return action according to an epsilon-greedy exploration policy
    """
    return 0


# #### Question 3
#
# Implement the `update` function

def update(state, action, reward, next_state, done):
    """
    ** TO BE COMPLETED **
    """

    # add data to replay buffer
    if done:
        next_state = None
    replay_buffer.push(state, action, reward, next_state)

    if len(replay_buffer) < BATCH_SIZE:
        return np.inf

    # get batch
    transitions = replay_buffer.sample(BATCH_SIZE)

    # Compute loss - TO BE IMPLEMENTED!
    values  = torch.zeros(BATCH_SIZE)   # to be computed using batch
    targets = torch.zeros(BATCH_SIZE)   # to be computed using batch
    loss = objective(values, targets)

    # Optimize the model - UNCOMMENT!
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

    return loss.detach().numpy()


# #### Question 4
# Train a DQN on the `env` environment.

EVAL_EVERY = 5
REWARD_THRESHOLD = 199

def train():
    state = env.reset()
    epsilon = EPSILON_START
    ep = 0
    total_time = 0
    while ep < N_EPISODES:
        action = choose_action(state, epsilon)

        # take action and update replay buffer and networks
        next_state, reward, done, _ = env.step(action)
        loss = update(state, action, reward, next_state, done)

        # update state
        state = next_state

        # end episode if done
        if done:
            state = env.reset()
            ep   += 1
            if ( (ep+1)% EVAL_EVERY == 0):
                rewards = eval_dqn()
                print("episode =", ep+1, ", reward = ", np.mean(rewards))
                if np.mean(rewards) >= REWARD_THRESHOLD:
                    break

            # update target network
            if ep % UPDATE_TARGET_EVERY == 0:
                target_net.load_state_dict(q_net.state_dict())
            # decrease epsilon
            epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) *                             np.exp(-1. * ep / DECREASE_EPSILON )

        total_time += 1

# Run the training loop
train()

# Evaluate the final policy
rewards = eval_dqn(20)
print("")
print("mean reward after training = ", np.mean(rewards))


# #### Question 5
#
# Experiment the policy network.

def show_video():
    html = []
    for mp4 in Path("videos").glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))

env = Monitor(env, './videos', force=True, video_callable=lambda episode: True)

for episode in range(1):
    done = False
    state = env.reset()
    while not done:
        action = choose_action(state, 0.0)
        state, reward, done, info = env.step(action)
env.close()
# show_video()


# ### Experiments: Do It Yourself

# Remember the set of global parameters:
# ```
# # Environment
# env = gym.make("CartPole-v0")
#
# # Discount factor
# GAMMA = 0.99
#
# # Batch size
# BATCH_SIZE = 256
# # Capacity of the replay buffer
# BUFFER_CAPACITY = 16384 # 10000
# # Update target net every ... episodes
# UPDATE_TARGET_EVERY = 32 # 20
#
# # Initial value of epsilon
# EPSILON_START = 1.0
# # Parameter to decrease epsilon
# DECREASE_EPSILON = 200
# # Minimum value of epislon
# EPSILON_MIN = 0.05
#
# # Number of training episodes
# N_EPISODES = 200
#
# # Learning rate
# LEARNING_RATE = 0.1
# ```

# #### Question 6
#
# Craft an experiment and study the influence of the `BUFFER_CAPACITY` on the learning process (speed of *convergence*, training curves...)

# #### Question 7
#
# Craft an experiment and study the influence of the `UPDATE_TARGET_EVERY` on the learning process (speed of *convergence*, training curves...)

# #### Question 8
#
# If you have the computer power to do so, try to do a grid search on those two hyper-parameters and comment the results. Otherwise, study the influence of another hyper-parameter.

# ## Bonus: SAIL-DQN
#
#
# `choose_action`, `get_q` and `eval_dqn` remain the same.
#
# To be implemented:
# * `update_sail`, compared to `update`, modifies $y_i$ as explained above.
# * `train_sail` adds several steps to `train`.
#
# Tip #1: `replay_buffer` now contains returns as well.
#
# Tip #2: in the computed advantage, use $Q(s_i, a_i, \theta^-)$, not $Q(s_i, a_i)$. It makes the bonus more stable.
#
# Tip #3: `torch.maximum` can be used to compute the element-wise max between two arrays.

# #### Question 9
#
# Implement `update_sail` function.

def update_sail(state, action, reward, next_state, done):
   """
   ** TO BE COMPLETED **
   """

   # add data to temporary replay buffer
   if done:
       next_state = None
   replay_buffer_temp.push(state, action, reward, next_state)

   if len(replay_buffer) < BATCH_SIZE:
       return np.inf

   # get batch
   transitions = replay_buffer.sample(BATCH_SIZE)

   # Compute loss - TO BE IMPLEMENTED!
   values  = torch.zeros(BATCH_SIZE)   # to be computed using batch
   targets = torch.zeros(BATCH_SIZE)   # to be computed using batch
   loss = objective(values, targets)

   # Optimize the model - UNCOMMENT!
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

   return loss.detach().numpy()


# #### Question 10
#
# Implement the training loop.

def get_episode_returns(rewards):
    returns_reversed = accumulate(rewards[::-1],
                                lambda x, y: x*GAMMA + y)
    return list(returns_reversed)[::-1]

def train_sail():
    state = env.reset()
    epsilon = EPSILON_START
    ep = 0
    total_time = 0
    while ep < N_EPISODES:
        action = choose_action(state, epsilon)

        # take action and update replay buffer and networks
        next_state, reward, done, _ = env.step(action)
        loss = update_sail(state, action, reward, next_state, done)

        # update state
        state = next_state

        # end episode if done
        if done:
            state = env.reset()
            ep   += 1
            if ( (ep+1)% EVAL_EVERY == 0):
                rewards = eval_dqn()
                print("episode =", ep+1, ", reward = ", np.mean(rewards))
                if np.mean(rewards) >= REWARD_THRESHOLD:
                    break

            # fetch transitions from the temporary memory
            transitions = replay_buffer_temp.memory

            # calculate episode returns
            # TO IMPLEMENT

            # transfer transitions completed with returns to main memory
            # TO IMPLEMENT

            # reset the temporary memory
            # TO IMPLEMENT

            # update target network
            if ep % UPDATE_TARGET_EVERY == 0:
                target_net.load_state_dict(q_net.state_dict())
            # decrease epsilon
            epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) *                             np.exp(-1. * ep / DECREASE_EPSILON )

        total_time += 1

# Run the training loop
train_sail()

# Evaluate the final policy
rewards = eval_dqn(20)
print("")
print("mean reward after training = ", np.mean(rewards))


# #### Question 11
#
# Display your policy in action.

from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
from IPython.display import clear_output
from pathlib import Path
import base64

def show_video():
    html = []
    for mp4 in Path("videos").glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))

env = Monitor(env, './videos', force=True, video_callable=lambda episode: True)

for episode in range(1):
    done = False
    state = env.reset()
    while not done:
        action = choose_action(state, 0.0)
        state, reward, done, info = env.step(action)
env.close()
# show_video()
