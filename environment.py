import numpy as np
from collections import namedtuple
import tensorflow as tf
from tensorflow import keras

# Custom time step structure
TimeStep = namedtuple('TimeStep', ['observation', 'reward', 'is_last'])

class SimpleConnect4Environment:
    def __init__(self, auto_opponent=True):
        self._rows = 6
        self._cols = 7
        self._num_actions = 7
        self._board = np.zeros((self._rows, self._cols), dtype=np.int32)
        self._current_player = 1 # 1 for player, -1 for opponent
        self._episode_ended = False
        self._auto_opponent = auto_opponent  # Whether to automatically make opponent moves
        
    def reset(self):
        self._board = np.zeros((self._rows, self._cols), dtype=np.int32)
        self._current_player = 1
        self._episode_ended = False
        return self._get_observation()
    
    def step(self, action):
        if self._episode_ended:
            return self.reset()
        
        # Safety check for invalid action
        if self._board[0, action] != 0:
            valid_actions = [i for i, valid in enumerate(self._get_valid_actions_mask()) if valid]
            if valid_actions:
                action = valid_actions[0]  # Take first valid action if possible
            else:
                self._episode_ended = True
                return TimeStep(self._get_observation(), 0.0, True)
        
        # Place piece on board
        for row in range(self._rows - 1, -1, -1):
            if self._board[row, action] == 0:
                self._board[row, action] = self._current_player
                break
        
        # Check for win
        if self._check_win(self._current_player):
            self._episode_ended = True
            return TimeStep(self._get_observation(), 10.0, True)
        
        # Check for draw
        if np.all(self._board != 0):
            self._episode_ended = True
            return TimeStep(self._get_observation(), 0.0, True)
        
        # Only make opponent move if auto_opponent is True
        if self._auto_opponent:
            # Opponent move (random)
            self._current_player = -self._current_player
            opponent_action = self._get_random_valid_action()
            if opponent_action is not None:
                for row in range(self._rows - 1, -1, -1):
                    if self._board[row, opponent_action] == 0:
                        self._board[row, opponent_action] = self._current_player
                        break
                
                if self._check_win(self._current_player):
                    self._episode_ended = True
                    return TimeStep(self._get_observation(), -10.0, True)
                
                if np.all(self._board != 0):
                    self._episode_ended = True
                    return TimeStep(self._get_observation(), 0.0, True)
            
            self._current_player = -self._current_player
        
        return TimeStep(self._get_observation(), 0.1, False)
    
    def _get_valid_actions_mask(self):
        """Returns a boolean array of valid moves (columns to place in)"""
        return np.array([self._board[0, col] == 0 for col in range(self._cols)], dtype=np.bool_)
    
    def _get_observation(self):
        """Get the current observation"""
        action_mask = self._get_valid_actions_mask()
        return {
            'board': self._board.copy(),
            'action_mask': action_mask
        }
    
    def _get_random_valid_action(self):
        """Get a random valid action for the opponent"""
        valid_actions = [col for col in range(self._cols) if self._board[0, col] == 0]
        if valid_actions:
            return np.random.choice(valid_actions)
        return None
    
    def _check_win(self, player):
        """Check if the given player has won"""
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
    
    def render(self):
        """Render the current board state"""
        print("\n" + "="*29)
        for row in range(6):
            print("|", end=" ")
            for col in range(7):
                if self._board[row, col] == 0:
                    print(".", end=" | ")
                elif self._board[row, col] == 1:
                    print("X", end=" | ")
                else:
                    print("O", end=" | ")
            print()
        print("="*29)
        print("  0   1   2   3   4   5   6  ")

    def _create_q_network(self):
        """Create the Q-network that handles the Connect4 observation structure"""
        board_input = keras.layers.Input(shape=(6, 7), name='board')
        board_flat = keras.layers.Flatten()(board_input)
        action_mask_input = keras.layers.Input(shape=(7,), name='action_mask')
        combined_input = keras.layers.Concatenate()([board_flat, action_mask_input])
        x = keras.layers.Dense(128, activation='relu')(combined_input)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        q_values = keras.layers.Dense(7, activation=None)(x)
        masked_q_values = keras.layers.Lambda(
            lambda x: tf.where(x[1], x[0], tf.constant(-1e6, dtype=tf.float32)),
            output_shape=(7,)
        )([q_values, action_mask_input])
        return keras.Model(
            inputs=[board_input, action_mask_input],
            outputs=masked_q_values
        )


if __name__ == "__main__":
    env = SimpleConnect4Environment()
    
    # Play a few random moves
    state = env.reset()
    env.render()
    
    for i in range(5):
        valid_actions = [i for i, valid in enumerate(state['action_mask']) if valid]
        action = np.random.choice(valid_actions)
        print(f"\nPlayer places in column {action}")
        
        time_step = env.step(action)
        state = time_step.observation
        env.render()
        
        if time_step.is_last:
            print(f"Game ended with reward: {time_step.reward}")
            break 