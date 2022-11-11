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
        self.grid_positions = [self.env.what_position(i) for i in self.observations]
        self.data = dict()
        while len(self.grid_positions) != len(set(self.grid_positions)):
            self.observations = [self.env.reset() for i in range(self.n_agents)]
            self.grid_positions = [self.env.what_position(i) for i in self.observations]

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
           # self.atualizar_flag_all_agents()
            states = self.env.get_states(observation_)
            self.data[i] = [self.observations[i]] + list(states)
            self.reward[i] = reward
            self.done[i] = done
        return self.observations, [self.env.what_position(a) for a in self.observations], self.reward, self.done
    
    def step2(self):
        for agent in range(self.n_agents):
            observation = self.observations[agent]
            self.env.set_state(observation)
            available_actions = self.env.available_action(observation)
            action = self.main_agent.choose_best_action(observation, available_actions)
            print('act', action)
            observation_, reward, done = self.env.step(action)

            self.env.set_state(observation_)
            dynamic, flag, drop, pick, gp = self.env.get_states(observation_)
            if flag == 2 and gp < self.env.col * self.env.col:
                    if len(self.book) > 0:
                        temporary_state = self.book.pop()
                        self.env.current_flag = 0
                        self.env.current_drop_off = temporary_state[0]
                        self.env.current_pick_up = temporary_state[1]
                        observation_ = self.env.att_state(gp)
            
            self.observations[agent] = observation_
            states = self.env.get_states(observation_)
            self.grid_positions[agent] = gp
            self.data[agent] = [observation_]+ list(states)
            self.done[agent] = done
            self.reward[agent] = reward
            self.atualizar_flag_all_agents()
            a = [self.env.get_states(i) for i in self.observations]
            for aaa, item in enumerate(a):
                print(self.observations[aaa], item)
        return self.observations, [self.env.what_position(a) for a in self.observations], self.reward, self.done
    
    def step_agents2(self, episode, maxEpisode):
        actions = np.zeros(self.n_agents, dtype=np.uint8)
        for agent in range(self.n_agents):
            observation = self.observations[agent]
            self.env.set_state(observation)
            available_action = self.env.available_action(observation)
            action = self.main_agent.choose_action(observation, episode, maxEpisode, available_action)
            actions[agent] = action
            observation_, reward, done = self.env.step(action)

            self.main_agent.learn(observation, action, reward, observation_, done)

            self.env.set_state(observation_)
            dynamic, flag, drop, pick, gp = self.env.get_states(observation_)
            if flag == 2 and gp < self.env.col * self.env.col:
                    if len(self.book) > 0:
                        temporary_state = self.book.pop()
                        self.env.current_flag = 0
                        self.env.current_drop_off = temporary_state[0]
                        self.env.current_pick_up = temporary_state[1]
                        observation_ = self.env.att_state(gp)
            
            self.observations[agent] = observation_
            states = self.env.get_states(observation_)
            self.grid_positions[agent] = gp
            self.data[agent] = [observation_]+ list(states)
            self.done[agent] = done
            self.reward[agent] = reward
            self.atualizar_flag_all_agents()
        info = {'grid_position': [self.env.what_position(a) for a in self.observations],
                'action': actions}
        return self.observations, self.reward, self.done, info
        


    
    def books(self, n):
        self.book = self.env.generate_demand(n)
    
    def atualizar_flag_all_agents(self):
        observations_copy = self.observations
        for idx, observation in enumerate(observations_copy):
            flag = [0,0,0,0]
            self.env.set_state(observation)
            available_action = self.env.available_action(observation)
            if 4 in available_action:
                available_action = np.setdiff1d(available_action, 4)
            for act in available_action:
                if self.env.move(act) in self.grid_positions:
                    flag[act] = 1
          #  possible_states = [(act, self.env.move(act)) for act in available_action]
          #  other_obs = [self.env.what_position(state) for state in np.setdiff1d(self.observations, observation)]
          #  for act, state_ in possible_states:
          #      if state_ in other_obs:
          #          flag[act] = 1
            flag = self.env.binary2decimal(''.join(map(str, flag)))
            self.data[idx][1] = flag
            self.env.current_dynamic, self.env.current_flag, \
            self.env.current_drop_off, self.env.current_pick_up,\
            self.env.grid_position = self.data[idx][1:]
            self.observations[idx] = self.data[idx][0] = self.env.att_state(self.data[idx][-1])
                   

if __name__ == '__main__':
    #env = GridWorld(9,9,-1,5,10,150,1)
    #env.set_pick_up([2, 3, 4, 5, 6])
    #env.set_drop_off([18, 25, 27, 30, 34, 39, 43, 48, 110, 113, 119, 122, 133, 142, 145])
    #env.set_obstacles([19, 20, 22, 23, 26, 28, 29, 31, 32, 35, 37, 38, 40, 41, 44, \
    #                   46, 47, 49, 50, 53, 90, 91, 93, 94, 97, 98, 99, 100, 102, \
    #                   103, 106, 107, 108, 109, 111, 112, 115, 116, 117, 118, 120, \
    #                   121,124, 125, 149, 150, 151, 152, 153, 154, 155, 156, 158, 159,\
    #                   160, 161])
    #env.possible_states()
    #env.load_available_action2()
    #env.load_available_flag_dynamic2()

    env = GridWorld(6, 6, -1, 50, 100, 150, 1)
    env.set_pick_up([1, 2, 3, 4])
    env.set_drop_off([55, 61, 58, 64])
    env.set_obstacles([13, 16, 19, 22, 54, 56, 57, 59, 60, 62, 63, 65])
    env.possible_states()
    env.load_available_action2()
    env.load_available_flag_dynamic2()
    agent = brain(.1, .99, .1, len(env.action_space()), len(env.state_space()))
    #agent.filter_q_table(env.state_action)
    agent.load('biblioteca66.txt')
    


    for all_estagios in range(8, 9):
        env.set_stage(5)
        if all_estagios <= 5:
            n_agents = 1
            n_epoch = 5000
            n_books = 0
            max_ep = n_epoch // 2
            env.set_stage(all_estagios)
        elif all_estagios == 6:
            n_agents = 2
            n_books = 10
            n_epoch = 100
            agent.epsilon = .2
            max_ep = 1
        elif all_estagios == 7:
            n_agents = 3
            n_epoch = 100
            n_books = 50
            agent.epsilon = .2
            max_ep = 1
        
        elif all_estagios == 8:
            print('estagio 8')
            n_agents = 6
            n_epoch = 1000
            n_books = 10
            agent.epsilon = .2
            max_ep = 1


        ma = multi_agent(agent, env, n_agents)
        reward_sum = np.zeros((n_agents, n_epoch))
        obstacle = env.obstacles
        points_obstacles = [np.array((state2cartesian(state))) for state in obstacle]

        drop_off = env.drop_off
        drop_off_points = [np.array((state2cartesian(state))) for state in drop_off]

        pick_up = env.pick_up
        pick_up_point = [np.array((state2cartesian(state))) for state in pick_up]
        #
        ma.reset()
        for epoch in range(n_epoch):
            observations = ma.reset()
            ma.books(n_books)
        #  img = np.zeros((300, 600, 3), dtype='uint8')
            done = [False, False]
            while not (True in done):
        #     cv2.imshow('Grid_World', img)
        #     cv2.waitKey(1)
        #     img = np.zeros((300, 600, 3), dtype='uint8')
        #  #      # Desenhar elementos estaticos
        #     for point in points_obstacles:
        #         cv2.rectangle(img, point, point + 50, (0, 0, 255), 5)
        # #    
        #     for point in drop_off_points :
        #         cv2.rectangle(img, point, point + 50, (0, 255, 255), 5)
        # #    #cv2.rectangle(img, drop_point, drop_point + 50, (0, 255, 255), -1)
        #     
        #     for point in pick_up_point:
        #         cv2.rectangle(img, point, point + 50, (0, 255, 0), 5)
        # # #   #cv2.rectangle(img, pick_point, pick_point + 50, (0, 255, 0), -1)
                observation_, reward, done, info = ma.step_agents2(epoch + 1, max_ep)
        #     agent_position = info['grid_position']
        #     print(reward)
                reward_sum[0][epoch] +=  reward[0]
            # view()

            # Takes step after fixed time
            #    t_end = time.time() + .5
            #    while time.time() < t_end:
            #        continue
                
            #    for idx, n_agnt in enumerate(agent_position):
            #        agent_state = n_agnt
            #        agent_point = np.array(state2cartesian(agent_state))
            #        cv2.rectangle(img, agent_point, agent_point + 50, [255, int(idx/2 * 255), idx*100], 3)

                
            print(epoch, end='\r')
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