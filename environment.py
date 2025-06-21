import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.policies import random_tf_policy
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

class Connect4Environment(py_environment.PyEnvironment):
    
    def __init__(self):
        self._rows = 6
        self._cols = 7
        self._num_actions = 7
        self._board = np.zeros((self._rows, self._cols), dtype=np.int32)
        self._current_player = 1
        self._episode_ended = False
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=6, name='action')
        
        self._observation_spec = {
            'board': array_spec.BoundedArraySpec(
                shape=(self._rows, self._cols), dtype=np.int32, 
                minimum=-1, maximum=1, name='board'),
            'action_mask': array_spec.BoundedArraySpec(
                shape=(self._num_actions,), dtype=np.bool_, 
                minimum=False, maximum=True, name='action_mask')
        }
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _get_valid_actions_mask(self):
        return np.array([self._board[0, col] == 0 for col in range(self._cols)], dtype=np.bool_)
    
    def _get_observation(self):
        action_mask = self._get_valid_action_mask()
        return {
            'board': self._board.copy(),
            'action_mask': action_mask
            }
    
    def _reset(self):
        self._board = np.zeros((self._rows, self._cols), dtype=np.int32)
        self._current_player = 1
        self._episode_ended = False
        return ts.restart(self._get_observation()) # Creates a TS signalling start of a new episode
    
    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        
        # Safety check in case agent somehow gives invalid action (maybe can remove this?)
        if self._board[0, action] != 0:
            valid_actions = [i for i, valid in enumerate(self._get_valid_actions_mask()) if valid]
            if valid_actions:
                action = valid_actions[0]  # Take first valid action if possible
            else:
                self._episode_ended = True
                return ts.termination(self._get_observation(), 0.0)
        
        # Place piece on board
        for row in range(self._rows - 1, -1, -1):
            if self._board[row, action] == 0:
                self._board[row, action] = self._current_player
                break
        
        # Check for win or draw
        if self._check_win(self._current_player):
            self._episode_ended = True
            return ts.termination(self._get_observation(), 10.0)
        
        if np.all(self._board != 0):
            self._episode_ended = True
            return ts.termination(self._get_observation(), 0.0)
        
        # Opponent move (same logic as before)
        self._current_player = -self._current_player
        opponent_action = self._get_random_valid_action()
        if opponent_action is not None:
            for row in range(self._rows - 1, -1, -1):
                if self._board[row, opponent_action] == 0:
                    self._board[row, opponent_action] = self._current_player
                    break
            
            if self._check_win(self._current_player):
                self._episode_ended = True
                return ts.termination(self._get_observation(), -10.0)
            
            if np.all(self._board != 0):
                self._episode_ended = True
                return ts.termination(self._get_observation(), 0.0)
        
        self._current_player = -self._current_player
        return ts.transition(self._get_observation(), 0.1)
    
    def _check_win(self, player):

        # Check horizontal
        for row in range(self._rows):
            for col in range(self._cols - 3):
                if all(self._board[row, col + i] == player for i in range(4)):
                    return True
        
        # Check vertical
        for row in range(self._rows - 3):
            for col in range(self._cols):
                if all(self._board[row + i, col] == player for i in range(4)):
                    return True
        
        # Check diagonal (positive slope)
        for row in range(self._rows - 3):
            for col in range(self._cols - 3):
                if all(self._board[row + i, col + i] == player for i in range(4)):
                    return True
        
        # Check diagonal (negative slope)
        for row in range(3, self._rows):
            for col in range(self._cols - 3):
                if all(self._board[row - i, col + i] == player for i in range(4)):
                    return True
        
        return False
    
    def _get_random_valid_action(self):
        """Get a random valid action for the opponent."""
        valid_actions = [col for col in range(self._cols) if self._board[0, col] == 0]
        if valid_actions:
            return np.random.choice(valid_actions)
        return None