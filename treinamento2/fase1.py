from multi_agent import *


def visualizar():
    obstacle = env.obstacles
    points_obstacles = [np.array((state2cartesian(state))) for state in obstacle]
    drop_off = env.drop_off
    drop_off_points = [np.array((state2cartesian(state))) for state in drop_off]
    pick_up = env.pick_up
    pick_up_point = [np.array((state2cartesian(state))) for state in pick_up]

    current_pick_up = ma.data[0][-2]
    pick_point = np.array(state2cartesian(pick_up[current_pick_up]))
    current_drop_off = ma.data[0][-3]
    drop_point = np.array(state2cartesian(drop_off[current_drop_off]))
    #img = np.zeros((450, 900, 3), dtype='uint8')

    
    img = np.zeros((450, 900, 3), dtype='uint8')
##      # Desenhar elementos estaticos
    for point in points_obstacles:
        cv2.rectangle(img, point, point + 50, (0, 0, 255), 5)
    for point in drop_off_points :
        cv2.rectangle(img, point, point + 50, (0, 255, 255), 5)
    cv2.rectangle(img, drop_point, drop_point + 50, (0, 255, 255), -1)
    for point in pick_up_point:
        cv2.rectangle(img, point, point + 50, (0, 255, 0), 5)
    cv2.rectangle(img, pick_point, pick_point + 50, (0, 255, 0), -1)

     #Takes step after fixed time
    t_end = time.time()
    while time.time() < t_end:
        continue

    agent_position = info['grid_position']

    for idx, n_agnt in enumerate(agent_position):
        agent_state = n_agnt
        agent_point = np.array(state2cartesian(agent_state))
        cv2.rectangle(img, agent_point, agent_point + 50, [255, int(idx/2 * 255), idx*100], 3)
    
    cv2.imshow('Grid_World', img)
    cv2.waitKey(1)
    pass


env = GridWorld(9, 9, -1, 50, 100, 150, 1)
env.set_pick_up([2, 3, 4, 5, 6])
env.set_drop_off([18, 25, 27, 30, 34, 39, 43, 48, 110, 113, 119, 122, 133, 142, 145])
env.set_obstacles([19, 20, 22, 23, 26, 28, 29, 31, 32, 35, 37, 38, 40, 41, 44, \
                    46, 47, 49, 50, 53, 90, 91, 93, 94, 97, 98, 99, 100, 102, \
                    103, 106, 107, 108, 109, 111, 112, 115, 116, 117, 118, 120, \
                    121, 124, 125, 149, 150, 151, 152, 153, 154, 155, 156, 158, 159,\
                    160, 161, 83, 87])
env.possible_states()
env.load_available_action2()
env.load_available_flag_dynamic2()

agent = brain(.1, .99, .2, len(env.action_space()), len(env.state_space()))
#agent.load('qtable.txt')

n_agents = 1
ma = multi_agent(agent, env, n_agents)

control_trainning = {0: {'epoch': [50, 100, 150, 200, 250, 300],
                        'epsilon': .1,
                        'retrain': [200, 300, 400, 1000, 1200, 1300],
                        'n_agents': 1,
                        'n_books': 0,
                        'max_ep': lambda x: x}}

for all_estagios in range(0, 6):
    print('\n', all_estagios)
    if all_estagios == 0:
        env.set_stage(0)
        n_epoch = control_trainning[0]['epoch'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios % 6 )
        ma.set_ep(.1, 1 - (all_estagios % 6) / 10, max_ep) #np.log10(10-all_estagios)
    
    elif all_estagios > 0 and all_estagios < 6:
        env.set_stage(0)
        n_epoch = control_trainning[0]['epoch'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios % 6)
        ma.set_ep(.1, 1 - (all_estagios % 6) / 10, max_ep) #np.log10(10-all_estagios)
    '''
    env.set_stage(1)
    if all_estagios < 6:
        n_epoch = control_trainning[0]['epoch'][all_estagios]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios)
        ma.set_ep(.1, 1 - all_estagios/10, max_ep) #np.log10(10-all_estagios)
    
    elif all_estagios == 6:
        transfer_learning_kevin(env, agent, 1)
        ma.load('qtable.txt')
        n_epoch = control_trainning[0]['retrain'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios)
        ma.set_ep(.1, 1 - (all_estagios % 6)/10, max_ep) #np.log10(10-all_estagios)
    
    elif all_estagios > 6 and all_estagios < 12:
        n_epoch = control_trainning[0]['retrain'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios)
        ma.set_ep(.1, 1 - (all_estagios % 6)/10, max_ep) #np.log10(10-all_estagios)
    
    elif all_estagios >= 12 and all_estagios < 18:
        n_epoch = control_trainning[0]['epoch'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios)
        ma.set_ep(.1, 1 - (all_estagios % 6)/10, max_ep) #np.log10(10-all_estagios)
    
    elif all_estagios == 18:
        transfer_learning_kevin(env, agent, 2)
        ma.load('qtable.txt')
        n_epoch = control_trainning[0]['retrain'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios)
        ma.set_ep(.1, 1 - (all_estagios % 6)/10, max_ep) #np.log10(10-all_estagios)
    
    elif all_estagios > 18 and all_estagios < 24:
        n_epoch = control_trainning[0]['retrain'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios)
        ma.set_ep(.1, 1 - (all_estagios % 6)/10, max_ep) #np.log10(10-all_estagios)
    
    elif all_estagios == 24:
        env.set_stage(0)
        #transfer_learning_kevin(env, agent, 3)
        #transfer_learning_kevin(env, agent, 4)
        #ma.load('qtable.txt')
        n_epoch = control_trainning[0]['epoch'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios % 6 )
        ma.set_ep(.1, 1 - (all_estagios % 6) / 10, max_ep) #np.log10(10-all_estagios)
    
    elif all_estagios > 24 and all_estagios < 30:
        env.set_stage(0)
        n_epoch = control_trainning[0]['epoch'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios % 6)
        ma.set_ep(.1, 1 - (all_estagios % 6) / 10, max_ep) #np.log10(10-all_estagios)
    
    elif all_estagios == 30:
        env.set_stage(0)
        transfer_learning_kevin(env, agent, 5)
        ma.load('qtable')
        n_epoch = control_trainning[0]['retrain'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios % 12)
        ma.set_ep(.1, 1 - (all_estagios % 6) / 10, max_ep) #np.log10(10-all_estagios)
    
    elif all_estagios > 30 and all_estagios < 36:
        env.set_stage(0)
        n_epoch = control_trainning[0]['retrain'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios % 12)
        ma.set_ep(.1, 1 - (all_estagios % 6) / 10, max_ep) #np.log10(10-all_estagios)
    
    elif all_estagios == 36:
        env.set_stage(2)
        transfer_learning_kevin(env, agent, 6)
        transfer_learning_kevin(env, agent, 7)
        ma.load('qtable')
        n_epoch = control_trainning[0]['retrain'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios % 12)
        ma.set_ep(.1, 1 - (all_estagios % 6) / 10, max_ep) #np.log10(10-all_estagios)
    
    elif all_estagios > 36 and all_estagios < 42:
        env.set_stage(2)
        n_epoch = control_trainning[0]['retrain'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios % 12)
        ma.set_ep(.1, 1 - (all_estagios % 6) / 10, max_ep) #np.log10(10-all_estagios)
    
    else:
        #ma.load('qtable')
        env.set_stage(2)
        n_epoch = control_trainning[0]['retrain'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = 2#control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(0)
        ma.set_ep(.1, .1, 1) #np.log10(10-all_estagios)
    '''


        
    
    ma.set_n_agents(n_agents)
    reward_sum = np.zeros((n_agents, n_epoch))

    for epoch in range(n_epoch):
        observations = ma.reset()
        #print(env.get_states(observations[0]))
        ma.books(n_books)
        done = [False, False]
       # ma.set_ep(.001, .1, 2)
        while not (True in done):
            observation_, reward, done, info = ma.step_agents2(epoch + 1, max_ep)
            #print(env.get_states(observation_[0]), reward, ma.main_agent.epsilon)
            reward_sum[0][epoch] +=  reward[0]
            visualizar()
        print(epoch, end='\r')
    print('len q table:', len(ma.get_q_table()))
    ma.save('qtable')
    plt.figure()
    plt.plot(reward_sum[0])
    plt.savefig(f'grafico{all_estagios}.png', dpi=600)
    plt.close()

