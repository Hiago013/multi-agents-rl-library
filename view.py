import numpy as np
import cv2
import time
from brain import brain
from GridWorld import GridWorld
from multi_agent import multi_agent


def state2cartesian(state):
    x, y = divmod(state, 9)
    return x * 50, y * 50

def cartesian2state(cartesian_point):
    x, y = cartesian_point
    x = x // 50
    y = y // 50
    return 9 * x + y
#####

env = GridWorld(9, 9, -1, 50, 100,150,1)
env.set_pick_up([2, 3, 4, 5, 6])
env.set_drop_off([18, 25, 27, 30, 34, 39, 43, 48, 110, 113, 119, 122, 133, 142, 145])
env.set_obstacles([19, 20, 22, 23, 26, 28, 29, 31, 32, 35, 37, 38, 40, 41, 44, \
                    46, 47, 49, 50, 53, 90, 91, 93, 94, 97, 98, 99, 100, 102, \
                    103, 106, 107, 108, 109, 111, 112, 115, 116, 117, 118, 120, \
                    121,124, 125, 149, 150, 151, 152, 153, 154, 155, 156, 158, 159,\
                    160, 161])
env.possible_states()
env.load_available_action2()
env.load_available_flag_dynamic2()
agent = brain(.1, .99, .1, len(env.action_space()), len(env.state_space()))
agent.filter_q_table(env.state_action)
agent.load('qtable.txt')
env.set_stage(1)
env.set_progressive_curriculum(5)
obstacle = env.obstacles
points_obstacles = [np.array((state2cartesian(state))) for state in obstacle]

drop_off = env.drop_off
drop_off_points = [np.array((state2cartesian(state))) for state in drop_off]

pick_up = env.pick_up
pick_up_point = [np.array((state2cartesian(state))) for state in pick_up]


#observation = env.reset()
n_agents = 1
ma = multi_agent(agent, env, n_agents)
color_agents = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for i in range(n_agents)]
agent_position = [1, 2]

print(agent.get_q_table()[24669])
print(agent.get_q_table()[24669])
print(env.get_states(24669))


for w in range(50):
    ma.books(0)
    print(ma.book)
    observations = ma.reset()
    current_pick_up = ma.data[0][-2]
    pick_point = np.array(state2cartesian(pick_up[current_pick_up]))
    #ma.observations = [811, 751]
    #observations = [811, 751]

    agent_positions = [ma.data[i][-1] for i in range(n_agents)]
    agent_point = [np.array(state2cartesian(agent_position)) for agent_position in agent_positions]


    crrnt_pckp_stt = pick_up[env.current_pick_up]
    crrnt_drpff_stt = drop_off[env.current_drop_off]

    #agent_position = env.grid_position
    #agent_point = np.array(state2cartesian(agent_position))
    #pick_point = np.array(state2cartesian(crrnt_pckp_stt))
    drop_point = np.array(state2cartesian(crrnt_drpff_stt))


    

    img = np.zeros((450, 900, 3), dtype='uint8')
    done = [False]
    while True:
        

        cv2.imshow('GridWorld', img)
        cv2.waitKey(1)
        img = np.zeros((450, 900, 3), dtype='uint8')
        # Desenhar elementos estaticos
        for point in points_obstacles:
            cv2.rectangle(img, point, point + 50, (0, 0, 255), 5)
        
        for point in drop_off_points :
            cv2.rectangle(img, point, point + 50, (0, 255, 255), 5)
        cv2.rectangle(img, drop_point, drop_point + 50, (0, 255, 255), -1)
        
        for point in pick_up_point:
            cv2.rectangle(img, point, point + 50, (0, 255, 0), 5)
        cv2.rectangle(img, pick_point, pick_point + 50, (0, 255, 0), -1)
        
        ##########

        if True in done:
            break

        if len(set(agent_position)) < len(agent_position):
            print('bateeeeeeeeu')
           # print(env.get_states(observations[0]))
           # print(env.get_states(observations[1]))
            while True:
                pass
        observations, agent_position, reward, done = ma.step2()
        print(' ')
        print(reward, observations, agent_position, agent.get_q_table()[observations[0]])
        print(' ')
        print('---')
        print(env.get_states(observations[0]))
        #print(env.get_states(observations[1]))
        #print(env.get_states(observations[2]))
        print('---')

        #print(agent_position)
        
       # print(observations, reward, [env.get_states(t)[0] for t in observations])
       # print(env.get_states(observations[0]))

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
        t_end = time.time() +1
        while time.time() < t_end:
            continue
        
        for idx, n_agnt in enumerate(agent_position):
            agent_state = n_agnt
            agent_point = np.array(state2cartesian(agent_state))
            cv2.rectangle(img, agent_point, agent_point + 50, color_agents[idx], 3)

        


        
cv2.destroyAllWindows()
