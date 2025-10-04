import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
    def choose_action(self, state):
        """Epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state])
        
    def update_epsilon(self):
        """Giảm epsilon sau mỗi episode"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    
    def learn(self, state, action, reward, next_state, gamma=0.95):
        """Cập nhật Q-table"""
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        new_value = old_value + self.lr * (reward + gamma * next_max - old_value)
        self.q_table[state, action] = new_value
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay