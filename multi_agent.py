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
        self.stack_stay = [Stack() for i in range(n_agents)]

    def set_q_table(self, q_table:np.array):
        self.q_table = q_table
    
    def get_q_table(self):
        return self.q_table
    
    def save(self, filename):
        if not '.txt' in filename:
            filename += '.txt'
        np.savetxt(filename, self.q_table)
    
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
            available_actions = self.env.available_action(observation)
            action = self.main_agent.choose_action(observation, episode, maxEpisode, available_actions)
            actions[i] = action

            #if action == 4:
            #    self.stack_stay[i].push(1)
            #else:
            #    self.stack_stay[i].pop()

            observation_, reward, done = self.env.step(action)

            #if len(self.stack_stay[i]) >= 4:
            #    reward -= 5

            # learn
            self.main_agent.learn(observation, action, reward, observation_, done)

            dynamic, flag, drop, pick, gp = self.env.get_states(observation_)
            if flag == 2 and gp < self.env.col * self.env.col:
                if len(self.book) > 0:
                    temporary_state = self.book.pop()
                    self.env.current_flag = 0
                    self.env.current_drop_off = temporary_state[0]
                    self.env.current_pick_up =temporary_state[1]
                    observation_ = self.env.att_state(gp)


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

            dynamic, flag, drop, pick, gp = self.env.get_states(observation_)
            if flag == 2 and gp < self.env.col * self.env.col:
                if len(self.book) > 0:
                    temporary_state = self.book.pop()
                    self.env.current_flag = 0
                    self.env.current_drop_off = temporary_state[0]
                    self.env.current_pick_up = temporary_state[1]
                    observation_ = self.env.att_state(gp)

            states = self.env.get_states(observation_)
            self.data[i] = [self.observations[i]] + list(states)
            self.observations[i] = self._att_flag(i, observation_)
            self.reward[i] = reward
            self.done[i] = done
        
       # print(self.data)
        return self.observations, [self.env.what_position(a) for a in self.observations], self.reward, self.done
    
    def books(self, n):
        self.book = self.env.generate_demand(n)

    

if __name__ == '__main__':
    env = GridWorld(5, 5, -1, 150, 150, 150, 1)
    env.set_pick_up([1, 2, 3])
    env.set_drop_off([35, 39])
    env.set_obstacles([0, 4, 5, 9, 10, 14, 15, 19, 20, 24, 36, 38, 41, 43])
    env.possible_states()
    env.load_available_action2()
    env.load_available_flag_dynamic2()
    agent = brain(.1, .99, .1, len(env.action_space()), len(env.state_space()))
    agent.filter_q_table(env.state_action)
    agent.load('qtablelib.txt')
    env.set_stage(5)

    n_agents = 2
    ma = multi_agent(agent, env, n_agents)
    n_epoch = 1000
    reward_sum = np.zeros((n_agents, n_epoch))
    print(reward_sum)
    print(reward_sum[0])
    for j in range(n_epoch):
        observations = ma.reset()
        ma.books(3)
        done = [False, False]
        while not (True in done):
            observation_, reward, done, info = ma.step_agents(j, n_epoch//2)
            reward_sum[0][j] +=  reward[0]
    ma.save('qtablelib')
    plt.plot(reward_sum[0])
   # plt.plot(reward_sum[1])
    
    plt.show()

    
