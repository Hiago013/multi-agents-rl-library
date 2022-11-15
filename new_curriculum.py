import numpy as np
from stack import Stack


class new_curriculum:
    def __init__(self, states:np.array, obstacles:list, elevator, col:int, row:int, pick_up:list, drop_off:list):
        self.states = states
        self.obstacles = obstacles
        self.col = col
        self.row = row
        self.axis_grid_position, self.axis_pick_up,\
        self.axis_drop_off, self.axis_flag, self.axis_dynamic = 4, 3, 2, 1, 0
        self.elevator = elevator
        self.stage = {}
        self.drop_off = drop_off
        self.pick_up = pick_up
        self.load_stages()

    
    def load_stages(self):
        all_grid_positions = np.arange(self.col * self.row * 2)
        # Primeiro Estagio
        # Vale apenas (0, 0, 0, pick, grid_position)
        states = np.array_split(all_grid_positions, 6)
        stages_aux = []
        del_drop_off = np.arange(1, len(self.drop_off))
        for idx, item in enumerate(states):
            del_grid_position = list(set(self.obstacles + list(np.setdiff1d(all_grid_positions, item))))
            aux = np.delete(self.states, del_grid_position, axis=self.axis_grid_position)
            aux = np.delete(aux, (1,2), axis = self.axis_flag)
            aux = np.delete(aux, np.arange(1,16), axis=self.axis_dynamic)
            #aux = np.delete(aux, del_drop_off, axis=self.axis_drop_off)  # posso tirar depois
            #aux = np.delete(aux, [0, 1, 3, 4], axis = self.axis_pick_up) # posso tirar depois
            stages_aux.append((idx, aux.flatten()))
        self.stage[0] = dict(stages_aux)

        # Segundo Estagio
        # Vale apenas (0, 1, drop, 0, [2, 3, 4, 5, 6])
        states = np.array_split(np.arange(len(self.drop_off)), 3)
        stages_aux = []
        del_pick_up = [0,1,3,4]
        for idx, item in enumerate(states):
            if idx<2:
                del_grid_position = np.setdiff1d(all_grid_positions, self.pick_up)
                del_drop_off = np.setdiff1d(states, item)
                aux = np.delete(self.states, del_grid_position, axis=self.axis_grid_position)
                aux = np.delete(aux, (0, 2), axis=self.axis_flag)
                aux = np.delete(aux, np.arange(1, 16), axis=self.axis_dynamic)
                aux = np.delete(aux, del_drop_off, axis=self.axis_drop_off)
                aux = np.delete(aux, del_pick_up, axis = self.axis_pick_up) # posso tirar depois
                stages_aux.append((idx, aux.flatten()))

        last_drop = item
        states_last_drop = np.array_split(all_grid_positions, 6)[::-1]
        for idx, item in enumerate(states_last_drop):
            del_grid_position = list(set(self.obstacles + list(np.setdiff1d(all_grid_positions, item))))
            del_drop_off = np.arange(len(self.drop_off)-1)
            #del_drop_off = np.setdiff1d(states, last_drop)
            aux = np.delete(self.states, del_grid_position, axis=self.axis_grid_position)
            aux = np.delete(aux, (0, 2), axis=self.axis_flag)
            aux = np.delete(aux, np.arange(1, 16), axis=self.axis_dynamic)
            aux = np.delete(aux, del_drop_off, axis=self.axis_drop_off)
            aux = np.delete(aux, del_pick_up, axis = self.axis_pick_up) # posso tirar depois
            stages_aux.append((idx+2, aux.flatten()))
        
        self.stage[1] = dict(stages_aux)

        # Terceiro Estagio
        states = np.arange(self.col * self.row)
        del_grid_position = list(set(self.obstacles + list(np.setdiff1d(all_grid_positions, states))))
        del_flag = (1, 2)
        aux = np.delete(self.states, del_grid_position, axis = self.axis_grid_position)
        aux = np.delete(aux, del_flag, axis=self.axis_flag)
        aux = np.delete(aux, np.arange(1, 16), axis=self.axis_dynamic)
        self.stage[2] = aux.flatten()
    
    def get_stage(self, stage):
        return self.stage[stage]
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from brain import brain
    from GridWorld import GridWorld

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
    #agent.load('qtable.txt')

    crr = new_curriculum(env.all_states, env.obstacles, env.elevator, 9, 9, env.pick_up, env.drop_off)
    crr.load_stages()
    print(((crr.stage[1][7])))