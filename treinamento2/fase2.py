from multi_agent import *


env = GridWorld(9, 9, -5, 50, 100, 150,1)
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
agent.load('qtable.txt')
transfer_learning(env, agent, 1)
agent.load('qtable.txt')
n_agents = 1
ma = multi_agent(agent, env, n_agents)

control_trainning = {0: {'epoch': [500, 400, 300, 200, 100, 50],
                        'retrain':[500, 200, 100, 100, 100, 100],
                        'epsilon': .1,
                        'n_agents': 1,
                        'n_books': 0,
                        'max_ep': lambda x: x//2}}
for all_estagios in range(6, 12):
    print('\n', all_estagios)
    env.set_stage(0)
    if all_estagios < 12:
        n_epoch = control_trainning[0]['retrain'][all_estagios - 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios)

    ma.set_n_agents(n_agents)
    reward_sum = np.zeros((n_agents, n_epoch))
    ma.reset()    
    for epoch in range(n_epoch):
        observations = ma.reset()
        ma.books(n_books)
        done = [False, False]
        while not (True in done):
            observation_, reward, done, info = ma.step_agents2(epoch + 1, max_ep)
            reward_sum[0][epoch] +=  reward[0]
        print(epoch, end='\r')
    ma.save('qtable')
    plt.plot(reward_sum[0])
    plt.show()


