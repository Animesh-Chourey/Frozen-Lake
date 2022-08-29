import numpy as np
from LinearWrapper import LinearWrapper
from chooseAction import choose_action
from frozenlake import FrozenLake
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        # TODO:
        q = features.dot(theta) # get value of q for given epsiode.
        done = False
        while not done:
            # select an action based on epsilon greedy policy
            if random_state.rand() < epsilon[i]:
                action = random_state.choice(env.n_actions)
            else:
                qmax = max(q)
                best = [a for a in range(env.n_actions) if np.allclose(qmax, q[a])]
                action = random_state.choice(best)
            #observed reward and feature matrix for the action
            next_features, r, done = env.step(action)
            delta = r - q[action]

            q = next_features.dot(theta)
            

            delta += gamma * max(q)
            # update value of theta using bellman equcation
            theta += eta[i] * delta * features[action, :]
            
            features = next_features

    return theta
