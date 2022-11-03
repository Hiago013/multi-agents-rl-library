from time import time
import matplotlib.pyplot as plt
import numpy as np
from brain import brain
from GridWorld import GridWorld

env = GridWorld(5, 5, -1, 5, 10, 100, 1)
env.set_pick_up([1, 2, 3])
env.set_drop_off([35, 39])
env.set_obstacles([0, 4, 5, 9, 10, 14, 15, 19, 20, 24, 36, 38, 41, 43])
env.possible_states()
env.load_available_action()
env.load_available_flag_dynamic()
agent = brain(.1, .99, .1, len(env.action_space()), len(env.state_space()))
agent.filter_q_table(env.state_action)
agent.load('qtable.txt')
num_epochs = 24000
score = np.zeros(num_epochs)
sum_score = 0

for crr in range(0,6):
  env.set_stage(crr)
  init = time()
  num_epochs = 12000
  if crr in [2, 3, 5]:
    num_epochs = 24000
  for i in range(num_epochs):
    observation = env.reset()
    env.current_dynamic = 0
    observation = env.att_state(env.grid_position)
    done = False


    while not done:
            available_actions = env.available_action(observation)
            action = agent.choose_action(observation, i, num_epochs, available_actions)
            observation_, reward, done = env.step(action)
            agent.learn(observation, action, reward, observation_, done)
            observation = observation_

            # removendo outros agentes
            #env.current_dynamic = 0
            #observation = env.att_state(env.grid_position_)

            observation = env.att_dynamic(observation)
            #print(env.grid_position)
            
            sum_score += reward
            if done:
            # print('done -->', env.grid_position)
              score[i] = sum_score
              sum_score = 0
              # print("Finished after {} timesteps".format(t+1))
              break
  print(time() - init)
  agent.save('qtable.txt')
  plt.plot(score)
  plt.grid()
  plt.show()
  
#[Output For Mountain Car Cont Env:] 
#[-0.56252328  0.00184034]
#[-0.56081509  0.00170819] -0.00796802138459 False {}
#[Output For CartPole Env:]
#[ 0.1895078   0.55386028 -0.19064739 -1.03988221]
#[ 0.20058501  0.36171167 -0.21144503 -0.81259279] 1.0 True {}
#Finished after 52 timesteps
