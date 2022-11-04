import numpy as np
import cv2
import time
from brain import brain
from GridWorld import GridWorld
from multi_agent import multi_agent


def state2cartesian(state):
    y, x = divmod(state, 5)
    return x * 50, y * 50

def cartesian2state(cartesian_point):
    y, x = cartesian_point
    x = x // 50
    y = y // 50
    return 5 * x + y
#####

env = GridWorld(5, 5, -1, 5, 10, 100, 1)
env.set_pick_up([1, 2, 3])
env.set_drop_off([35, 39])
env.set_obstacles([0, 4, 5, 9, 10, 14, 15, 19, 20, 24, 36, 38, 41, 43])
env.possible_states()
env.load_available_action()
env.load_available_flag_dynamic()
agent = brain(.1, .99, .1, len(env.action_space()), len(env.state_space()))
agent.filter_q_table(env.state_action)
agent.load('qtable2.txt')
#env.set_stage(0)

obstacle = env.obstacles
points_obstacles = [np.array((state2cartesian(state))) for state in obstacle]

drop_off = env.drop_off
drop_off_points = [np.array((state2cartesian(state))) for state in drop_off]

pick_up = env.pick_up
pick_up_point = [np.array((state2cartesian(state))) for state in pick_up]


#observation = env.reset()
n_agents = 2
ma = multi_agent(agent, env, n_agents)
observations = ma.reset()

agent_positions = [ma.data[i][-1] for i in range(n_agents)]
agent_point = [np.array(state2cartesian(agent_position)) for agent_position in agent_positions]


#crrnt_pckp_stt = pick_up[env.current_pick_up]
#crrnt_drpff_stt = drop_off[env.current_drop_off]

#agent_position = env.grid_position
#agent_point = np.array(state2cartesian(agent_position))
#pick_point = np.array(state2cartesian(crrnt_pckp_stt))
#drop_point = np.array(state2cartesian(crrnt_drpff_stt))
img = np.zeros((500, 250, 3), dtype='uint8')
done = [False]
while True:

    cv2.imshow('a', img)
    cv2.waitKey(1)
    img = np.zeros((500, 250, 3), dtype='uint8')
    # Desenhar elementos estaticos
    for point in points_obstacles:
        cv2.rectangle(img, point, point + 50, (0, 0, 255), 5)
    
    for point in drop_off_points :
        cv2.rectangle(img, point, point + 50, (0, 255, 255), 5)
    #cv2.rectangle(img, drop_point, drop_point + 50, (0, 255, 255), -1)
    
    for point in pick_up_point:
        cv2.rectangle(img, point, point + 50, (0, 255, 0), 5)
    #cv2.rectangle(img, pick_point, pick_point + 50, (0, 255, 0), -1)
    
    ##########

    if not False in done:
        break


    observations, agent_position, reward, done = ma.step()
    print(reward)

    '''
    action = agent.choose_best_action(observation)
    observation_, reward, done = env.step(action)
    observation = observation_

    # removendo outros agentes
    env.current_dynamic = 0
    observation = env.att_state(env.grid_position_)
    '''

    #observation = env.att_dynamic(observation)
    #print(env.grid_position)
    
    #############

    # Takes step after fixed time
    t_end = time.time() + 0.5
    while time.time() < t_end:
        continue
    
    for idx, n_agnt in enumerate(agent_position):
        agent_state = n_agnt
        agent_point = np.array(state2cartesian(agent_state))
        cv2.rectangle(img, agent_point, agent_point + 50, [255, int(idx/2 * 255), idx*100], 3)


    
cv2.destroyAllWindows()
