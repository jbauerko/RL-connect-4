import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
import os
import time
from environment import SimpleConnect4Environment

class ActionMaskLayer(keras.layers.Layer):
    """Custom layer to apply action mask to Q-values"""
    
    def __init__(self, **kwargs):
        super(ActionMaskLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        q_values, action_mask = inputs
        # Apply action mask to set invalid actions to large negative values
        return tf.where(tf.cast(action_mask, tf.bool), q_values, tf.constant(-1e6, dtype=tf.float32))
    
    def get_config(self):
        config = super(ActionMaskLayer, self).get_config()
        return config

def mse_loss(y_true, y_pred):
    """Mean squared error loss function"""
    return tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred))

class SimpleConnect4Agent:
    def __init__(self, environment, learning_rate=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize the Connect4 DQN Agent
        
        Args:
            environment: The SimpleConnect4Environment instance
            learning_rate: Learning rate for the optimizer
            epsilon: Initial exploration rate
            epsilon_decay: Rate at which epsilon decays
            epsilon_min: Minimum exploration rate
        """
        self.environment = environment
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Create the neural network
        self.q_network = self._create_q_network()
        self.target_network = self._create_q_network()
        self.update_target_network()
        
        # Compile the model
        self.q_network.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=mse_loss
        )
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        
        # Training parameters
        self.batch_size = 16
        self.gamma = 0.95  # Discount factor
        
    def _create_q_network(self):
        """Create the Q-network that handles the Connect4 observation structure"""
        
        # Input for the board state (6x7 grid)
        board_input = keras.layers.Input(shape=(6, 7), name='board')
        
        # Flatten the board
        board_flat = keras.layers.Flatten()(board_input)
        
        # Input for action mask (7 boolean values)
        action_mask_input = keras.layers.Input(shape=(7,), name='action_mask')
        
        # Concatenate board and action mask
        combined_input = keras.layers.Concatenate()([board_flat, action_mask_input])
        
        # Simpler dense layers
        x = keras.layers.Dense(64, activation='relu')(combined_input)
        x = keras.layers.Dense(32, activation='relu')(x)
        
        # Output layer (7 actions)
        q_values = keras.layers.Dense(7, activation=None)(x)
        
        # Apply action mask to set invalid actions to large negative values
        masked_q_values = ActionMaskLayer()([q_values, action_mask_input])
        
        return keras.Model(
            inputs=[board_input, action_mask_input],
            outputs=masked_q_values
        )
    
    def update_target_network(self):
        """Update the target network with weights from the main network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_actions):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            # Random action from valid actions
            return random.choice(valid_actions)
        
        # Get Q-values for current state
        board = np.array([state['board']])
        action_mask = np.array([state['action_mask']])
        
        q_values = self.q_network.predict([board, action_mask], verbose=0)[0]
        
        # Choose action with highest Q-value among valid actions
        valid_q_values = [(i, q_values[i]) for i in valid_actions]
        best_action = max(valid_q_values, key=lambda x: x[1])[0]
        
        return best_action
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare training data
        boards = []
        action_masks = []
        targets = []
        
        for state, action, reward, next_state, done in batch:
            board = state['board']
            action_mask = state['action_mask']
            
            # Get current Q-values
            current_q_values = self.q_network.predict(
                [np.array([board]), np.array([action_mask])], verbose=0
            )[0]
            
            if done:
                # If episode is done, target is just the reward
                target = reward
            else:
                # Get next state Q-values from target network
                next_board = next_state['board']
                next_action_mask = next_state['action_mask']
                next_q_values = self.target_network.predict(
                    [np.array([next_board]), np.array([next_action_mask])], verbose=0
                )[0]
                # Target is reward + discounted max Q-value
                target = reward + self.gamma * np.max(next_q_values)
            
            # Update Q-value for the action taken
            current_q_values[action] = target
            
            boards.append(board)
            action_masks.append(action_mask)
            targets.append(current_q_values)
        
        # Only train if we have valid data
        if len(boards) > 0:
            # Train the network
            self.q_network.fit(
                [np.array(boards), np.array(action_masks)],
                np.array(targets),
                epochs=1,
                verbose=0
            )
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, episodes=1000, max_steps=100):
        """Train the agent"""
        print(f"Starting training for {episodes} episodes...")
        
        for episode in range(episodes):
            state = self.environment.reset()
            total_reward = 0
            
            for step in range(max_steps):
                # Get valid actions
                valid_actions = [i for i, valid in enumerate(state['action_mask']) if valid]
                
                if not valid_actions:
                    print(f"Episode {episode}: No valid actions at step {step}")
                    break
                
                # Choose action
                action = self.act(state, valid_actions)
                
                # Take action
                next_state = self.environment.step(action)
                reward = next_state.reward
                done = next_state.is_last
                
                # Store experience
                self.remember(state, action, reward, next_state.observation, done)
                
                # Train on batch
                self.replay()
                
                state = next_state.observation
                total_reward += reward
                
                if done:
                    print(f"Episode {episode}: Game ended at step {step} with reward {reward}")
                    break
            
            # Update target network every 10 episodes
            if episode % 10 == 0:
                self.update_target_network()
            
            # Print progress
            if episode % 100 == 0:
                print(f"Episode {episode}/{episodes}, Epsilon: {self.epsilon:.3f}, Total Reward: {total_reward}")
        
        print("Training completed!")
    
    def evaluate(self, episodes=10):
        """Evaluate the agent's performance"""
        print(f"Evaluating agent over {episodes} episodes...")
        
        total_rewards = []
        wins = 0
        
        for episode in range(episodes):
            state = self.environment.reset()
            total_reward = 0
            
            while True:
                # Get valid actions
                valid_actions = [i for i, valid in enumerate(state['action_mask']) if valid]
                
                if not valid_actions:
                    break
                
                # Choose best action (no exploration)
                board = np.array([state['board']])
                action_mask = np.array([state['action_mask']])
                q_values = self.q_network.predict([board, action_mask], verbose=0)[0]
                
                valid_q_values = [(i, q_values[i]) for i in valid_actions]
                action = max(valid_q_values, key=lambda x: x[1])[0]
                
                # Take action
                next_state = self.environment.step(action)
                total_reward += next_state.reward
                state = next_state.observation
                
                if next_state.is_last:
                    break
            
            total_rewards.append(total_reward)
            if total_reward > 0:  # Win
                wins += 1
        
        avg_reward = np.mean(total_rewards)
        win_rate = wins / episodes
        
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Win Rate: {win_rate:.2f} ({wins}/{episodes})")
        
        return avg_reward, win_rate
    
    def play_game(self, render=True):
        """Play a complete game using the trained agent"""
        state = self.environment.reset()
        total_reward = 0
        step_count = 0
        
        while True:
            if render:
                self._render_board(state['board'])
            
            # Get valid actions
            valid_actions = [i for i, valid in enumerate(state['action_mask']) if valid]
            
            if not valid_actions:
                break
            
            # Choose best action
            board = np.array([state['board']])
            action_mask = np.array([state['action_mask']])
            q_values = self.q_network.predict([board, action_mask], verbose=0)[0]
            
            valid_q_values = [(i, q_values[i]) for i in valid_actions]
            action = max(valid_q_values, key=lambda x: x[1])[0]
            
            # Take action
            next_state = self.environment.step(action)
            total_reward += next_state.reward
            step_count += 1
            
            state = next_state.observation
            
            if next_state.is_last:
                break
        
        if render:
            self._render_board(state['board'])
            print(f"Game finished! Total reward: {total_reward}, Steps: {step_count}")
        
        return total_reward, step_count
    
    def save_model(self, filepath):
        """Save the trained agent"""
        os.makedirs(filepath, exist_ok=True)
        self.q_network.save(os.path.join(filepath, 'q_network.keras'))
        
        # Save training parameters
        params = {
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min
        }
        np.save(os.path.join(filepath, 'params.npy'), params)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained agent"""
        self.q_network = keras.models.load_model(os.path.join(filepath, 'q_network.keras'))
        self.target_network = keras.models.load_model(os.path.join(filepath, 'q_network.keras'))
        
        # Load training parameters
        params = np.load(os.path.join(filepath, 'params.npy'), allow_pickle=True).item()
        self.epsilon = params['epsilon']
        self.learning_rate = params['learning_rate']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon_min = params['epsilon_min']
        
        print(f"Model loaded from {filepath}")
    
    def _render_board(self, board):
        """Render the Connect4 board"""
        print("\n" + "="*29)
        for row in range(6):
            print("|", end=" ")
            for col in range(7):
                if board[row, col] == 0:
                    print(".", end=" | ")
                elif board[row, col] == 1:
                    print("X", end=" | ")
                else:
                    print("O", end=" | ")
            print()
        print("="*29)
        print("  0   1   2   3   4   5   6  ")


if __name__ == "__main__":
    # Example usage
    from environment import SimpleConnect4Environment
    
    # Create environment and agent
    env = SimpleConnect4Environment()
    agent = SimpleConnect4Agent(env)
    
    # Train the agent
    print("Starting training...")
    agent.train(episodes=50)
    
    # Evaluate the agent
    print("\nEvaluating trained agent...")
    agent.evaluate(episodes=20)
    
    # Play a game
    print("\nPlaying a game with trained agent...")
    agent.play_game()
    
    # Save the model
    agent.save_model("trained_agent") 