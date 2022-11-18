import numpy as np
from brain import brain
from GridWorld import GridWorld
class transfer(object):
    def __init__(self):
        pass

    def from_to(self, agent : brain, state, state_):
        agent.set_qvalue(state_, agent.get_q_table()[state])
        return agent

if __name__ == '__main__':
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
    agent.load('qtable.txt')


    # # #Primeiro Estagio
    # # train_states = dict()
    # # aux = []
    # # for gp in env.get_possibles_grid_positions():
    # #    for pick in range(len(env.pick_up)):
    # #        for drop in range(len(env.drop_off)):
    # #            if gp not in {2, 3, 5, 6}:
    # #                aux.append(env.get_observation((0, 0, drop, pick, gp)))
    # #    train_states[env.get_observation((0, 0, 0, 2, gp))] = aux
    # #    aux = []
    # # #Transferencia do conhecimento do primeiro estagio
    # # transfer_learning = transfer()
    # # for key in train_states.keys():
    # #    for state in train_states[key]:
    # #        agent = transfer_learning.from_to(agent, state = key, state_ = state) 
    # # agent.save('qtable2.txt')

    # # # Primeiro Estagio
    # # train_states = dict()
    # # aux = []
    # # for gp in env.get_possibles_grid_positions():
    # #     for pick in range(len(env.pick_up)):
    # #         for drop in range(len(env.drop_off)):
    # #             aux.append(env.get_observation((0, 0, drop, pick, gp)))
    # #         train_states[env.get_observation((0, 0, 0, pick, gp))] = aux
    # #         aux = []
    # # # Transferencia do conhecimento do primeiro estagio
    # # transfer_learning = transfer()
    # # for key in train_states.keys():
    # #     for state in train_states[key]:
    # #         agent = transfer_learning.from_to(agent, state = key, state_ = state) 
    # # agent.save('qtable3.txt')

    agent.load('qtable.txt')
    # Chegar corretamento no pick para qualquer drop
    # Primeiro Estagio
    train_states = dict()
    aux = []
    for gp in env.get_possibles_grid_positions():
        for pick in range(len(env.pick_up)):
            for drop in range(len(env.drop_off)):
                aux.append(env.get_observation((0, 2, drop, pick, gp)))
        train_states[env.get_observation((0, 0, 0, 2, gp))] = aux
        aux = []
    # Transferencia do conhecimento do primeiro estagio
    transfer_learning = transfer()
    for key in train_states.keys():
        for state in train_states[key]:
            agent = transfer_learning.from_to(agent, state = key, state_ = state) 
    agent.save('qtable4.txt')





   # # Primeiro Estagio - de Volta Pra Casa
   # train_states = dict()
   # aux = []

   # for gp in env.get_possibles_grid_positions():
   #     for pick in range(len(env.pick_up)):
   #         for drop in range(len(env.drop_off)):
   #             if gp not in {2, 3, 5, 6}:
   #                 aux.append(env.get_observation((0, 2, drop, pick, gp)))
   #     train_states[env.get_observation((0, 0, 0, 2, gp))] = aux
   #     aux = []
   # # Transferencia do conhecimento do primeiro estagio
   # transfer_learning = transfer()
   # for key in train_states.keys():
   #     for state in train_states[key]:
   #         agent = transfer_learning.from_to(agent, state = key, state_ = state)
   # 
   # agent.save('qtable.txt')

    




    '''
    train_states = dict()
    aux = []
    # Primeiro Estagio
    for pick in range(len(env.pick_up)):
        for gp in env.get_possibles_grid_positions():
            for drop in range(len(env.drop_off)):
                aux.append(env.get_observation((0, 0, drop, pick, gp)))
            train_states[env.get_observation((0, 0, 0, pick, gp))] = aux[1:]
            aux = []
    # Transferencia do conhecimento do primeiro estagio
    transfer_learning = transfer()
    for key in train_states.keys():
        for state in train_states[key]:
            agent = transfer_learning.from_to(agent, state = key, state_ = state)
    
    # Segundo Estagio
    # Vale apenas (0, 1, drop, 0, [2, 3, 4, 5, 6])
    train_states = dict()
    aux = []
    for drop in range(len(env.drop_off)):
        for gp in env.pick_up:
            for pick in range(len(env.pick_up)):
                aux.append(env.get_observation((0, 1, drop, pick, gp)))
            train_states[env.get_observation((0, 1, drop, 0, gp))] = aux[1:]

    
    agent.save('qtable2.txt')
    '''
