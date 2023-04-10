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
        remove_first_axis = set(np.arange(self.col * self.row, self.col * self.row * 2))
        remove_first_axis.update(set(self.obstacles))
        remove_first_axis.update(self.elevator)

        self.first_stage = np.delete(self.states, list(remove_first_axis), axis=self.axis_grid_position)
        self.first_stage = np.delete(self.first_stage, (1, 2), axis=self.axis_flag)
        self.first_stage = np.delete(self.first_stage, np.arange(1,16), axis=self.axis_dynamic)
        self.first_stage = np.delete(self.first_stage, np.arange(1, 15), axis=self.axis_drop_off)
        self.first_stage = self.first_stage.flatten()


        self.second_stage = np.delete(self.states, list(remove_first_axis), axis=self.axis_grid_position)
        self.second_stage = np.delete(self.second_stage, (0, 2), axis=self.axis_flag)
        self.second_stage = self.second_stage.flatten()


        remove_third_axis = set(np.arange(0, self.col * self.row))
        remove_third_axis.update(set(self.obstacles))

        self.third_stage = np.delete(self.states, list(remove_third_axis), axis = self.axis_grid_position)
        self.third_stage = np.delete(self.third_stage, (0, 2), axis = self.axis_flag)
        self.third_stage =  self.third_stage.flatten()

        self.forth_stage = np.delete(self.states, list(remove_third_axis), axis = self.axis_grid_position)
        self.forth_stage = np.delete(self.forth_stage, (0, 1), axis = self.axis_flag)
        self.forth_stage = self.forth_stage.flatten()


        self.fifth_stage = np.delete(self.states, list(remove_first_axis), axis=self.axis_grid_position)
        self.fifth_stage = np.delete(self.fifth_stage, (0, 1), axis=self.axis_flag)
        self.fifth_stage = self.fifth_stage.flatten()

        self.stages = {0:self.first_stage,
                       1:self.second_stage,
                       2:self.third_stage,
                       3:self.forth_stage,
                       4:self.fifth_stage}
    
    def get_stage(self, stage):
        return self.stages[stage]
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from brain import brain
    from GridWorld import GridWorld

    env = GridWorld(9, 9, -1, 50, 100, 150,1)
    env.set_pick_up([2, 3, 4, 5, 6])
    env.set_drop_off([18, 25, 27, 30, 34, 39, 43, 48, 110, 113, 119, 122, 133, 142, 145])
    env.set_obstacles([19, 20, 22, 23, 26, 28, 29, 31, 32, 35, 37, 38, 40, 41, 44, \
                       46, 47, 49, 50, 53, 90, 91, 93, 94, 97, 98, 99, 100, 102, \
                       103, 106, 107, 108, 109, 111, 112, 115, 116, 117, 118, 120, \
                       121,124, 125, 149, 150, 151, 152, 153, 154, 155, 156, 158, 159,\
                       160, 161])
    env.possible_states()
    #env.load_available_action2()
    #env.load_available_flag_dynamic2()

    #agent = brain(.1, .99, .1, len(env.action_space()), len(env.state_space()))
    #agent.load('qtable.txt')

    crr = new_curriculum(env.all_states, env.obstacles, env.elevator, 9, 9, env.pick_up, env.drop_off)

    #print((crr.stages[0]))
    #print((crr.stage[1][12]))
    for item in crr.stages[0]:
       print(env.get_states(item))
       print('')
    #    for state in crr.stage[1][key]:
    #        print(env.get_states(state))
    #    break
    #    print(' ')

   # print(((crr.stage[0])))