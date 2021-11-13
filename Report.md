
# Deep Reinforcement Navigation

## Introduction

The goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.


![Gif of my implementation](media/agent_working.gif)


## Environment


A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:


0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.


The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

The project environment is similar to, but not identical to the Banana Collector environment on the Unity ML-Agents.

## Model Architecture

### Deep Q Network

Reinforcement learning is a branch of machine learning where an agent outputs an action and the environment returns an observation or, the state of the system and a reward. The goal of the agent is to best decide the course of action. Usually RL is decribed in terms of this agent interacting with a previously known environment, trying to maximize the overall reward. 

The Deep Q-Learning algorithm represents the optimal action-value function *q as a neural network (instead of a table).  

It used two features:
> - Experience Replay
> - Fixed Q-Targets

There are two ways for Deep Q Algorithm: 

Method one: Use of rolly history of the past data via replay pool. By using replay pool, the behaviour distribution is average over many of its previous states, smoothing out learning and avoiding oscillations. Experience of each state is used in many weight updates. 

Method two: Use of target network to represent old Q-dunction, which will be used to compute the loss of every action during training. At each step of the training, the Q fuction value changes and the value estimate can spiral out of control. These additions enable RL changes to converge, more reliably during training.

![deep q network architecture](media/model_arch.png)


## Hyperparameters


> - BUFFER_SIZE = int(1e5)  # replay buffer size 
> - BATCH_SIZE = 64         # minibatch size
> - GAMMA = 0.99            # discount factor
> - TAU = 1e-3              # for soft update of target parameters
> - LR = 5e-4               # learning rate 
> - UPDATE_EVERY = 4        # how often to update the network


## Algorithm


![Deep Q algorithm](media/algorithm.png)

## Implementation

### model.py

QNetwork model is implemented using PyTorch. This model will be trained to predict the best action which must be taken. 

Structure of model:

1. Input layer whose size is taken from state_size
2. Two hidden layers with 64 cells
3. Output layer whose side depends on the action size

### dqn_agent.py

DQN agent and replay buffer are defined. 

DQN agent:

1. __init__ : state_size and action_size are defined. Adam Optimizer and QNetwork model are initialized. ReplayBuffer will be used to store the states, actions and rewards.
2. step: Memory is added in the ReplayBuffer. A random sample is taken from ReplayBuffer memory. Method learn is called to improve the model.
3. act: best action is chosen and returned 
4. learn: experience, targets, loss(mean squared error) and optimizer is used to update the weights for neural networks
5. soft_update: it is used to update target neural networks from local neural network

ReplayBuffer:

1. init: variables are initialized
2. add: experience are appended in memory
3. sample: a rando sample of action, reward, states are returned
4. len: length of memory is returned

### Navigation.ipynb

This is where the agent is trained.

1. Unity environment is initialized
2. dqn: DQN agent is trained using ReplayBuffer memory
3. Plot score vs episode
4. model is saved in checkpoint.pth

## Results

### Model Training


![agent model training](media/agent_score.png)


### Plots of Rewards

![plot of rewards](media/agent_training_plot.png)

## Ideas For Future Works

Improvments which can be made:

1. Double DQN

Deep Q-Learning tends to overestimate action values. Double Q-Learning has been shown to work well in practice to help with this.


2. Prioritized Experience Replay

Deep Q-Learning samples experience transitions uniformly from a replay memory. Prioritized experienced replay is based on the idea that the agent can learn more effectively from some transitions than from others, and the more important transitions should be sampled with higher probability.



3. Dueling DQN

Currently, in order to determine which states are (or are not) valuable, we have to estimate the corresponding action values for each action. However, by replacing the traditional Deep Q-Network (DQN) architecture with a dueling architecture, we can assess the value of each state, without having to learn the effect of each action.


Another implementation (as suggessted in the course) is using pixels instead of velocity, along with ray-based perception of objects around its forward direction. For thos method we need to create a convolution network to train the agent.

## References

1. Deep Reinforcement Learning Udacity Nanodegree