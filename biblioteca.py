from stack import Stack
from GridWorld import GridWorld
from brain import brain
import numpy as np
from matplotlib import pyplot as plt
from multi_agent import multi_agent
import cv2

def cartesian2state(cartesian_point):
    y, x = cartesian_point
    x = x // 50
    y = y // 50
    return 13 * x + y

def state2cartesian(state):
    x, y = divmod(state, 13)
    return x * 50, y * 50


env = GridWorld(13, 13, -1, 5, 10, 150, 1)
env.set_pick_up([3, 4, 5, 6, 7, 8, 9])
env.set_drop_off([208, 210, 212, 214, 216, 218, 220, 286, 288, 290, 292, 294, 296, 298])
env.set_obstacles([0, 12, 13, 25, 26, 38, 39, 51, 52, 64, 65, 77, 78, 90, 91, 103, 104,\
                   116, 117,129,130,142,143,155,156, 168, 169, 170, 171, 172, 174, 175,\
                   175, 176, 178, 179, 180, 181, 196, 198, 200, 202, 204, 206, 209, 211,\
                   213, 215, 217, 219, 222, 224, 226, 228, 230, 232, 274, 276, 278, 280,\
                   282, 284, 287, 289, 291, 293, 295, 297, 300, 302, 304, 306, 308, 310])

env.possible_states()
env.load_available_action2()
env.load_available_flag_dynamic2()
agent = brain(.1, .99, .1, len(env.action_space()), len(env.state_space()))
n_agents = 1
ma = multi_agent(agent, env, n_agents)
n_epoch = 1000
reward_sum = np.zeros((n_agents, n_epoch))
print(reward_sum)
print(reward_sum[0])
for j in range(n_epoch):
    observations = ma.reset()
    done = [False, False]
    while False in done:
        observation_, reward, done, info = ma.step_agents(j, n_epoch//2)
        reward_sum[0][j] +=  reward[0]
ma.save('qtable2')

plt.plot(reward_sum[0])
plt.show()


'''


obstacle = env.obstacles
points_obstacles = [np.array((state2cartesian(state))) for state in obstacle]

drop_off = env.drop_off
drop_off_points = [np.array((state2cartesian(state))) for state in drop_off]

pick_up = env.pick_up
pick_up_point = [np.array((state2cartesian(state))) for state in pick_up]


#observation = env.reset()
n_agents = 1
#ma = multi_agent(agent, env, n_agents)
#print(ma.get_q_table()[4258])
for w in range(10):
   # observations = ma.reset()
    #ma.observations = [811, 751]
    #observations = [811, 751]

    #agent_positions = [ma.data[i][-1] for i in range(n_agents)]
    #agent_point = [np.array(state2cartesian(agent_position)) for agent_position in agent_positions]


    #crrnt_pckp_stt = pick_up[env.current_pick_up]
    #crrnt_drpff_stt = drop_off[env.current_drop_off]

    #agent_position = env.grid_position
    #agent_point = np.array(state2cartesian(agent_position))
    #pick_point = np.array(state2cartesian(crrnt_pckp_stt))
    #drop_point = np.array(state2cartesian(crrnt_drpff_stt))

    img = np.zeros((650, 1300, 3), dtype='uint8')
    done = [False]
    while True:

        cv2.imshow('GridWorld', img)
        cv2.waitKey(1)
        img = np.zeros((650, 1300, 3), dtype='uint8')
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


        #observations, agent_position, reward, done = ma.step()
        #print(observations, reward, [env.get_states(t)[0] for t in observations])

        
        action = agent.choose_best_action(observation)
        observation_, reward, done = env.step(action)
        observation = observation_

        # removendo outros agentes
        env.current_dynamic = 0
        observation = env.att_state(env.grid_position_)
        

        #observation = env.att_dynamic(observation)
        #print(env.grid_position)
        
        #############

        # Takes step after fixed time
       # t_end = time.time() + 0.5
        #while time.time() < t_end:
        #    continue
        
        #for idx, n_agnt in enumerate(agent_position):
        #    agent_state = n_agnt
        #    agent_point = np.array(state2cartesian(agent_state))
        #    cv2.rectangle(img, agent_point, agent_point + 50, [255, int(idx/2 * 255), idx*100], 3)


        
    cv2.destroyAllWindows()






n_agents = 3
ma = multi_agent(agent, env, n_agents)
n_epoch = 5000
reward_sum = np.zeros((n_agents, n_epoch))
print(reward_sum)
print(reward_sum[0])
for j in range(n_epoch):
    observations = ma.reset()
    done = [False, False]
    while False in done:
        observation_, reward, done, info = ma.step_agents(j, n_epoch//2)
        reward_sum[0][j] +=  reward[0]
ma.save('qtable2')

plt.plot(reward_sum[0])
# plt.plot(reward_sum[1])

plt.show()

    
'''