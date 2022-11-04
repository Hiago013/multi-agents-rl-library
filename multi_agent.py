from stack import Stack
from GridWorld import GridWorld
from brain import brain
import numpy as np
from matplotlib import pyplot as plt

class multi_agent():
    def __init__(self, agent : brain, grid_world : GridWorld, n_agents : int):
        self.main_agent = agent
        self.env = grid_world
        self.n_agents = n_agents
        self.q_table = agent.get_q_table()
        self.reward = [0 for n_agnt in range(n_agents)]
        self.done = [False for n_agnt in range(n_agents)]

    def set_q_table(self, q_table:np.array):
        self.q_table = q_table
    
    def get_q_table(self):
        return self.q_table
    
    def save(self, filename):
        if not '.txt' in filename:
            filename += '.txt'
        np.savetxt(filename, self.q_table)

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

        for idx, obs in enumerate(self.observations):
            self.observations[idx] = self._att_flag(idx, obs)

        return self.observations
    
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
        # print(self.data)
        actions = [act for act in range(self.n_agents)]
        for i in range(self.n_agents):
            observation = self._att_flag(i, self.observations[i])
            self.env.set_state(observation)
            available_actions = env.available_action(observation)
            action = self.main_agent.choose_action(observation, episode, maxEpisode, available_actions)
            actions[i] = action
            #print(action)
            observation_, reward, done = self.env.step(action)

            # learn
            self.main_agent.learn(observation, action, reward, observation_, done)

            states = self.env.get_states(observation_)
            self.data[i] = [self.observations[i]] + list(states)
            self.observations[i] = self._att_flag(i, observation_)
            self.reward[i] = reward
            self.done[i] = done
        
        info = {'grid_position': [self.env.what_position(a) for a in self.observations],
                'action': actions}

        return self.observations, self.reward, self.done, info

    
    def step(self):
       # print(self.data)
        for i in range(self.n_agents):
            observation = self._att_flag(i, self.observations[i])
            self.env.set_state(observation)
            action = self.main_agent.choose_best_action(observation)
            #print(action)
            observation_, reward, done = self.env.step(action)
            states = self.env.get_states(observation_)
            self.data[i] = [self.observations[i]] + list(states)
            self.observations[i] = self._att_flag(i, observation_)
            self.reward[i] = reward
            self.done[i] = done
        
       # print(self.data)
        return self.observations, [self.env.what_position(a) for a in self.observations], self.reward, self.done

    

if __name__ == '__main__':
    env = GridWorld(5, 5, -1, 5, 10, 150, 1)
    env.set_pick_up([1, 2, 3])
    env.set_drop_off([35, 39])
    env.set_obstacles([0, 4, 5, 9, 10, 14, 15, 19, 20, 24, 36, 38, 41, 43])
    env.possible_states()
    env.load_available_action()
    env.load_available_flag_dynamic()
    agent = brain(.1, .99, .1, len(env.action_space()), len(env.state_space()))
    agent.filter_q_table(env.state_action)
    agent.load('qtable2.txt')
    env.set_stage(5)

    n_agents = 3
    ma = multi_agent(agent, env, n_agents)
    n_epoch = 2000
    reward_sum = np.zeros((n_agents, n_epoch))
    print(reward_sum)
    print(reward_sum[0])
    for j in range(n_epoch):
        observations = ma.reset()
        done = [False, False]
        while False in done:
            observation_, reward, done, info = ma.step_agents(j, n_epoch//2)
            reward_sum[0][j] +=  reward[0]
            #reward_sum[1][j] +=  reward[1]
        
            #for i in range(1):
                #print(observations)
                #observation = ma._att_flag(i, observations[i])
                #env.set_state(observation)
                #available_actions = env.available_action(observation)
                #action = agent.choose_action(observation, i, 2, available_actions)
                #observation_, reward, done = env.step(action)
                #agent.learn(observation, action, reward, observation_, done)
                #observations[i] = observation_
                #dones[i] = done
    #done = [False, False]
    #while False in done:
    #    a, b, c, done = ma.step()
    #    print(b)
    ma.save('qtable2')

    plt.plot(reward_sum[0])
   # plt.plot(reward_sum[1])
    
    plt.show()

    