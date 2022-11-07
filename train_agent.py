from stack import Stack
from GridWorld import GridWorld
from brain import brain
import numpy as np
from matplotlib import pyplot as plt
from multi_agent import multi_agent
import cv2

def create_grid_world():
    env = GridWorld(5, 5, -1, 5, 10, 150, 1)
    env.set_pick_up([1, 2, 3])
    env.set_drop_off([35, 39])
    env.set_obstacles([0, 4, 5, 9, 10, 14, 15, 19, 20, 24, 36, 38, 41, 43])
    env.possible_states()
    env.load_available_action2()
    env.load_available_flag_dynamic2()
    return env

def create_brain(n_agents : int, env : GridWorld):
    agent = brain(.1, .99, .1, len(env.action_space()), len(env.state_space()))
    agent.load('qtablelib.txt')
    ma = multi_agent(agent, env, n_agents)
    return agent, ma

def train(env: GridWorld, agents : multi_agent, curr : int, n_epoch : int, n_agents:int):
    if n_agents == 1:
        env.set_stage(curr)
        reward_sum = np.zeros((n_agents, n_epoch))
        for epoch in range(n_epoch):
            observations = agents.reset()
            done = [False, False]
            while False in done:
                observation_, reward, done, info = agents.step_agents(epoch, n_epoch)
                reward_sum[0][epoch] +=  reward[0]
    plt.plot(reward_sum[0])
    plt.show()
    agents.save('qtablelib')


def main():
    env = create_grid_world()
    main_agent, ma = create_brain(1, env)
    train(env, ma, 5, 1000, 1)
    #train(env, ma, 1, 1000, 1)
    #train(env, ma, 2, 1000, 1)
    #train(env, ma, 3, 1000, 1)
    #train(env, ma, 4, 1000, 1)

if __name__ == '__main__':
    main()

