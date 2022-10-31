import numpy as np
class brain():
  def __init__(self, epsilon, gamma, alpha, n_actions, n_states):
    self.epsilon = epsilon
    self.gamma = gamma
    self.alpha = alpha
    self.actions = np.arange(n_actions)
    self.q_table = np.zeros((n_states, n_actions))
    self.epsilonFunction = lambda episode, maxEp: (max(1 - 1/maxEp * episode, .1))
  
  def filter_q_table(self, state_action):
    for state in state_action.keys():
      for action in self.actions:
        if action not in state_action[state]:
          self.q_table[state][action] = -1000

  
  def choose_action(self, state, episode, maxEpisode, actions):
    if np.random.uniform() < self.epsilonFunction(episode, maxEpisode):
      return np.random.choice(actions)
    best_action = np.argmax(self.q_table[state,:])
    return best_action
  
  def choose_best_action(self, state):
    best_action = np.argmax(self.q_table[state,:])
    return best_action
  
  def learn(self, state, action, reward, state_, done):
    self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * (reward + (1-done) * self.gamma * np.max(self.q_table[state_,:]))