import numpy as np
import pickle

n_trajectories = 6666
def is_state_final(state):
    return state[0] >= 6 and state[1] >= 6


def get_next_state_from_action(state, action):

    if state[2] >= 2 and action!='rest':
        increment = [0, 0, 1]
        new_state = [state[i] + increment[i] for i in [0, 1, 2]]
        new_state[0] = np.min([np.max([new_state[0], 0]), 7])
        new_state[1] = np.min([np.max([new_state[1], 0]), 7])
        new_state[2] = np.min([np.max([new_state[2], 0]), 7])

        return new_state

    elif action == 'st1':
        increment = [1, 0, 1]

    elif action == 'st2':
        if state[2] >=1 :
            increment = [0, 0, 2]
        else:
            increment = [2, 0, 1]

    elif action == 'flx1':
        increment = [0, 1, 1]

    elif action == 'flx2':
        if state[2] >= 1:
            increment = [0, 0, 2]
        else:
            increment = [0, 2, 1]

    elif action == 'rest':
        increment = [0, 0, -2]

    new_state = [state[i] + increment[i] for i in [0,1,2]]
    new_state[0] = np.min([np.max([new_state[0], 0]), 7])
    new_state[1] = np.min([np.max([new_state[1], 0]), 7])
    new_state[2] = np.min([np.max([new_state[2], 0]), 7])

    return new_state

episode_states =[]
episode_actions = []
for n in range(n_trajectories):
    state_list = [[0, 0, 0]]
    action_list = []
    while not is_state_final(state_list[-1]):
        if state_list[-1][2] >= 4:
            action = 'rest'
        else:
            action = np.random.choice(['st1', 'st2', 'flx1', 'flx2', 'rest'])
        action_list.append(action)
        new_state = get_next_state_from_action(state_list[-1], action)
        state_list.append(new_state)
    episode_actions.append(action_list)
    episode_states.append(state_list)


filehandler = open('episode_actions.pkl', 'wb')
pickle.dump(episode_actions, filehandler)

filehandler = open('episode_states.pkl', 'wb')
pickle.dump(episode_states, filehandler)


