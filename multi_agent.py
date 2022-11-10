from stack import Stack
from GridWorld import GridWorld
from brain import brain
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time

def state2cartesian(state):
    x, y = divmod(state, 6)
    return x * 50, y * 50

def cartesian2state(cartesian_point):
    y, x = cartesian_point
    x = x // 50
    y = y // 50
    return 6 * x + y
#####

class multi_agent():
    def __init__(self, agent : brain, grid_world : GridWorld, n_agents : int):
        self.main_agent = agent
        self.env = grid_world
        self.n_agents = n_agents
        self.q_table = agent.get_q_table()
        self.reward = [0 for n_agnt in range(n_agents)]
        self.done = [False for n_agnt in range(n_agents)]
        self.stack_stay = [Stack() for i in range(n_agents)]

    def set_q_table(self, q_table:np.array):
        self.q_table = q_table
    
    def get_q_table(self):
        return self.q_table
    
    def save(self, filename):
        if not '.txt' in filename:
            filename += '.txt'
        np.savetxt(filename, self.main_agent.get_q_table())
    
    def load(self, filename):
        if not '.txt' in filename:
            filename += '.txt'
        self.set_q_table(np.loadtxt(filename))

    def reset(self):
        self.observations = [self.env.reset() for i in range(self.n_agents)]
        grid_positions = [self.env.what_position(i) for i in self.observations]
        self.data = dict()
        while len(grid_positions) != len(set(grid_positions)):
            self.observations = [self.env.reset() for i in range(self.n_agents)]
            grid_positions = [self.env.what_position(i) for i in self.observations]

        for i in range(self.n_agents):
            states = self.env.get_states(self.observations[i])
            self.data[i] = [self.observations[i]] + list(states)        
        self.atualizar_flag_all_agents()
        return self.observations
    
    def zero_flag(self, n_agent):
        self.data[n_agent][1] = 0
        self.env.current_dynamic, self.env.current_flag, \
        self.env.current_drop_off, self.env.current_pick_up,\
        self.env.grid_position = self.data[n_agent][1:]
        return self.env.att_state(self.data[n_agent][-1])


    
    def _att_flag(self, n_agent, observation):
        flag = [0,0,0,0]
        self.env.grid_position = self.env.what_position(observation)
        available_action = self.env.available_action(observation)[:-1]
        possibles_states = [(act, self.env.what_position(self.env.move(act))) for act in available_action]
        for i in range(self.n_agents):
            if i != n_agent:
                for act, state in possibles_states:
                    if self.data[i][-1] == state:
                        flag[act] = 1
        current_flag = self.env.binary2decimal(''.join(map(str, flag)))
        
        self.data[n_agent][1] = current_flag

        self.env.current_dynamic, self.env.current_flag, \
        self.env.current_drop_off, self.env.current_pick_up,\
        self.env.grid_position = self.data[n_agent][1:]
        return self.env.att_state(self.data[n_agent][-1])

    def step_agents(self, episode, maxEpisode):
        actions = [act for act in range(self.n_agents)]
        for i in range(self.n_agents):
            self.atualizar_flag_all_agents()
            observation = self.observations[i]
            self.env.set_state(observation)
            available_actions = self.env.available_action(observation)
            action = self.main_agent.choose_action(observation, episode, maxEpisode, available_actions)
            actions[i] = action

            observation_, reward, done = self.env.step(action)

            self.main_agent.learn(observation, action, reward, observation_, done)

            dynamic, flag, drop, pick, gp = self.env.get_states(observation_)
            if flag == 2 and gp < self.env.col * self.env.col:
                if len(self.book) > 0:
                    temporary_state = self.book.pop()
                    self.env.current_flag = 0
                    self.env.current_drop_off = temporary_state[0]
                    self.env.current_pick_up =temporary_state[1]
                    observation_ = self.env.att_state(gp)
            
            self.observations[i] = self.data[i][0] = observation_
            self.atualizar_flag_all_agents()
            states = self.env.get_states(observation_)
            self.data[i] = [self.observations[i]] + list(states)
            self.reward[i] = reward
            self.done[i] = done
        
        info = {'grid_position': [self.env.what_position(a) for a in self.observations],
                'action': actions}

        return self.observations, self.reward, self.done, info

    
    def step(self):
        for i in range(self.n_agents):
            self.atualizar_flag_all_agents()
            observation = self.observations[i]
            self.env.set_state(observation)
            available_actions = self.env.available_action(observation)
            action = self.main_agent.choose_best_action(observation)
            observation_, reward, done = self.env.step(action)
            
            dynamic, flag, drop, pick, gp = self.env.get_states(observation_)
            if flag == 2 and gp < self.env.col * self.env.col:
                if len(self.book) > 0:
                    temporary_state = self.book.pop()
                    self.env.current_flag = 0
                    self.env.current_drop_off = temporary_state[0]
                    self.env.current_pick_up = temporary_state[1]
                    observation_ = self.env.att_state(gp)

            self.observations[i] = self.data[i][0] = observation_
            self.atualizar_flag_all_agents()
            states = self.env.get_states(observation_)
            self.data[i] = [self.observations[i]] + list(states)
            self.reward[i] = reward
            self.done[i] = done
        return self.observations, [self.env.what_position(a) for a in self.observations], self.reward, self.done
    
    def books(self, n):
        self.book = self.env.generate_demand(n)
    
    def atualizar_flag_all_agents(self):
        for idx, observation in enumerate(self.observations):
            flag = [0,0,0,0]
            self.env.set_state(observation)
            available_action = self.env.available_action(observation)
            possible_states = [(act, self.env.move(act)) for act in available_action]
            other_obs = [self.env.what_position(state) for state in np.setdiff1d(self.observations, observation)]
            for act, state_ in possible_states:
                if state_ in other_obs:
                    flag[act] = 1
            flag = self.env.binary2decimal(''.join(map(str, flag)))
            self.data[idx][1] = flag
            self.env.current_dynamic, self.env.current_flag, \
            self.env.current_drop_off, self.env.current_pick_up,\
            self.env.grid_position = self.data[idx][1:]
            self.observations[idx] = self.data[idx][0] = self.env.att_state(self.data[idx][-1])
                
            

def view():
        cv2.imshow('a', img)
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

        # Takes step after fixed time
            t_end = time.time()
            while time.time() < t_end:
                continue
            
            for idx, n_agnt in enumerate(agent_position):
                agent_state = n_agnt
                agent_point = np.array(state2cartesian(agent_state))
                cv2.rectangle(img, agent_point, agent_point + 50, [255, int(idx/2 * 255), idx*100], 3)

    

if __name__ == '__main__':
    #env = GridWorld(5, 5, -1, 150, 150, 150, 1)
    #env.set_pick_up([1, 2, 3])
    #env.set_drop_off([35, 39])
    #env.set_obstacles([0, 4, 5, 9, 10, 14, 15, 19, 20, 24, 36, 38, 41, 43])
    #env = GridWorld(13, 13, -1, 5, 10, 150, 1)
    #env.set_pick_up([5, 6, 7])
    #env.set_drop_off([208, 212, 216, 220, 288, 292, 296])
    #env.set_obstacles([0, 12, 13, 25, 26, 38, 39, 51, 52, 64, 65, 77, 78, 90, 91, 103, 104,\
    #               116, 117,129,130,142,143,155,156, 168, 169, \
    #               181, 196, 198, 200, 202, 204, 206, 209, 211,\
    #               213, 215, 217, 219, 222, 224, 226, 228, 230, 232, 274, 276, 278, 280,\
    #               282, 284, 287, 289, 291, 293, 295, 297, 300, 302, 304, 306, 308, 310])
    env = GridWorld(6, 6, -1, 5, 10, 150, 1)
    env.set_pick_up([1, 2, 3, 4])
    env.set_drop_off([55, 61, 58, 64])
    env.set_obstacles([13, 16, 19, 22, 54, 56, 57, 59, 60, 62, 63, 65])
    env.possible_states()
    env.load_available_action2()
    env.load_available_flag_dynamic2()
    agent = brain(.1, .99, .1, len(env.action_space()), len(env.state_space()))
    agent.filter_q_table(env.state_action)
    #agent.load('biblioteca66.txt')
    env.set_stage(3)

    n_agents = 1
    ma = multi_agent(agent, env, n_agents)
    n_epoch = 2000
    reward_sum = np.zeros((n_agents, n_epoch))
    print(reward_sum)
    print(reward_sum[0])
    #
    obstacle = env.obstacles
    points_obstacles = [np.array((state2cartesian(state))) for state in obstacle]

    drop_off = env.drop_off
    drop_off_points = [np.array((state2cartesian(state))) for state in drop_off]

    pick_up = env.pick_up
    pick_up_point = [np.array((state2cartesian(state))) for state in pick_up]
    #
    ma.reset()
    for j in range(n_epoch):
        observations = ma.reset()
        ma.books(0)
        img = np.zeros((300, 600, 3), dtype='uint8')
        done = [False, False]
        while not (True in done):
            cv2.imshow('Grid_World', img)
            cv2.waitKey(1)
            img = np.zeros((300, 600, 3), dtype='uint8')
          #      # Desenhar elementos estaticos
            for point in points_obstacles:
                cv2.rectangle(img, point, point + 50, (0, 0, 255), 5)
        #    
            for point in drop_off_points :
                cv2.rectangle(img, point, point + 50, (0, 255, 255), 5)
        #    #cv2.rectangle(img, drop_point, drop_point + 50, (0, 255, 255), -1)
            
            for point in pick_up_point:
                cv2.rectangle(img, point, point + 50, (0, 255, 0), 5)
        # #   #cv2.rectangle(img, pick_point, pick_point + 50, (0, 255, 0), -1)
            observation_, reward, done, info = ma.step_agents(j, n_epoch//2)
            agent_position = info['grid_position']
           # print(reward)
            reward_sum[0][j] +=  reward[0]
           # view()

           # Takes step after fixed time
            t_end = time.time()
            while time.time() < t_end:
                continue
            
            for idx, n_agnt in enumerate(agent_position):
                agent_state = n_agnt
                agent_point = np.array(state2cartesian(agent_state))
                cv2.rectangle(img, agent_point, agent_point + 50, [255, int(idx/2 * 255), idx*100], 3)

            
        print(j, end='\r')
    ma.save('biblioteca66')
    plt.plot(reward_sum[0])
   # plt.plot(reward_sum[1])
    
    plt.show()




'''
    
    done = [False]
    while True:

        cv2.imshow('a', img)
        cv2.waitKey(1)
        img = np.zeros((250, 500, 3), dtype='uint8')
       
        
        ##########

        if not False in done:
            break


        observations, agent_position, reward, done = ma.step()
        print(observations, reward, [env.get_states(t)[0] for t in observations])

        
        action = agent.choose_best_action(observation)
        observation_, reward, done = env.step(action)
        observation = observation_

        # removendo outros agentes
        env.current_dynamic = 0
        observation = env.att_state(env.grid_position_)
        

        #observation = env.att_dynamic(observation)
        #print(env.grid_position)
        
        #############

        


        
    cv2.destroyAllWindows()

    
'''