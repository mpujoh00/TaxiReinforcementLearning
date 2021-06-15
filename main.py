import pickle
import random
import gym
import numpy as np

# default values
num_episodes = 2000
max_steps = 200
beta = 0.7  # learning rate
gamma = 0.6  # discount factor
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 0.01


# algorithm that trains a model
def train_model(visualize=False):
    global epsilon
    for episode in range(num_episodes):
        state = environment.reset()
        if visualize:
            print('Initial state')
            environment.render()
        total_reward = 0
        # steps
        for t in range(max_steps):
            ## chooses an action given a random number
            exploit_explore_tradeoff = random.uniform(0, 1)
            # exploitation (chooses the best action)
            if exploit_explore_tradeoff > epsilon:
                action = np.argmax(Q[state, :])
            # exploration (chooses a random action)
            else:
                action = environment.action_space.sample()

            ## performs the action and gets the reward
            new_state, reward, done, info = environment.step(action)
            # visualization
            if visualize:
                environment.render()
                print(f'Reward = {reward}')

            ## updates Q
            Q[state, action] = Q[state, action] + beta * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
            # update of total reward and state
            total_reward += reward
            state = new_state

            ## ending of the episode
            if done:  # passenger has reached its destination
                print('Passenger reached its destination!')
                break
        print(f'Total reward of episode {episode + 1}: {total_reward}')
        # reduces exploration probability (reducing epsilon)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)


print('What do you want to do?\n\t1. Train model.\n\t2. Visualize episode of trained model.')
option = int(input())

if option == 1:
    print('##### Training model #####\n')
    # loading and rendering taxi environment
    environment = gym.make('Taxi-v3').env
    environment.reset()
    # initialization of Q (500x6 table, 500 possible states, 6 different actions)
    Q = np.zeros((environment.observation_space.n, environment.action_space.n))
    # training
    train_model()
    # saves the model
    print('Do you want to save the trained model? (y/n)')
    answer = input()
    if answer == 'y':
        with open('trained_model.pkl', 'wb') as f:
            pickle.dump(
                [environment, Q, max_steps, beta, gamma, epsilon, max_epsilon, min_epsilon, decay], f)

elif option == 2:
    print('##### Visualizing episode #####\n')
    # loads trained model
    with open('trained_model.pkl', 'rb') as f:
        environment, Q, max_steps, beta, gamma, epsilon, max_epsilon, min_epsilon, decay = pickle.load(f)
    # visualizes 1 episode
    num_episodes = 1
    train_model(visualize=True)

else:
    print('Incorrect option.')
