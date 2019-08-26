import pickle
from collections import defaultdict
import numpy as np
from data_generation import get_next_state_from_action

with open('episode_actions.pkl', 'rb') as file:
    episode_actions = pickle.load(file)

with open('episode_states.pkl', 'rb') as file:
    episode_states = pickle.load(file)
n_episodes = len(episode_actions)


Q = defaultdict(lambda: -666)

alpha = 0.5
gama = 0.5

possible_actions = ['st1', 'st2', 'flx1', 'flx2', 'rest']

for i in [0, 1, 2, 3, 4, 5, 6]:
    for a in possible_actions:
        Q[(tuple([6, 6, i]), a)] = 0
        Q[(tuple([6, 7, i]), a)] = 0
        Q[(tuple([7, 6, i]), a)] = 0


def get_best_action(Q, state, possible_actions=possible_actions):
    vals=[]
    for action in possible_actions:
        vals.append(Q[(state, action)])
    vals = np.array(vals)
    a_idx = np.random.choice(np.flatnonzero(vals == vals.max()))
    return possible_actions[a_idx]

for i_episode in range(1, n_episodes):

    episode_st = episode_states[i_episode]
    episode_act= episode_actions[i_episode]

    # For each step in the episode, backwards
    T = len(episode_act)
    print('T', T)
    for t in range(T):
        state = tuple(episode_st[t])
        action = episode_act[t]
        next_state = tuple(episode_st[t+1])

        #print('state:', state, '   action:', action, '     next_state:', next_state)

        if next_state[0] >= 6 and next_state[1] >= 6:
            reward = 1
        else:
            reward = -1

        # Update the total reward since step t
        current_q = Q[(state, action)]
        best_next_val = Q[(next_state, get_best_action(Q=Q, state=next_state))]
        Q[(state, action)] = current_q + alpha*(reward + gama*best_next_val - current_q)
        x=1


possible_actions = ['st1', 'st2', 'flx1', 'flx2', 'rest']
initial_state = tuple([0,0,0])

def is_state_final(state):
    return state[0] >= 6 and state[1] >= 6

def simulate_episode(Q, initial_state=initial_state,
                     possible_actions=possible_actions):

    list_of_states = [initial_state]
    list_of_actions = []
    state=initial_state
    i=0
    while (not is_state_final(state) and i < 1000):
        best_action = get_best_action(Q, state, possible_actions)
        next_state = get_next_state_from_action(state, best_action)
        print('state:', state, '   best_action:', best_action, '     next_state:', next_state)
        state = tuple(next_state)
        i += 1
    print('episode steps', i)

simulate_episode(Q)