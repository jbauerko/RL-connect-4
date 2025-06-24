import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import os
import tensorflow as tf
from environment import SimpleConnect4Environment
from agent import SimpleConnect4Agent

class Connect4GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Connect4 vs RL Agent")
        self.root.geometry("800x600")
        
        # Game state
        self.env = SimpleConnect4Environment(auto_opponent=False)
        self.agent = None
        self.current_level = None
        self.game_active = False
        self.player_turn = True  # True for human player, False for AI
        
        # Colors
        self.colors = {
            'empty': '#FFFFFF',
            'player': '#FF6B6B', 
            'ai': '#F7DC6F',
            'grid': '#2C3E50',
            'text': '#000000',
            'card_bg': '#F0F0F0',
            'accent': '#FFD700'
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Connect4 vs RL Agent", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Training level selection
        ttk.Label(control_frame, text="Model:").grid(row=0, column=0, padx=(0, 10))
        self.level_var = tk.StringVar(value="beginner")
        level_combo = ttk.Combobox(control_frame, textvariable=self.level_var, 
                                  values=["beginner", "intermediate", "expert"], 
                                  state="readonly", width=15)
        level_combo.grid(row=0, column=1, padx=(0, 10))
        
        # Train button
        self.train_button = ttk.Button(control_frame, text="Train Model", 
                                      command=self.train_agent)
        self.train_button.grid(row=0, column=2, padx=(0, 10))
        
        # Load button
        self.load_button = ttk.Button(control_frame, text="Load Model", 
                                     command=self.load_agent)
        self.load_button.grid(row=0, column=3, padx=(0, 10))
        
        # New game button
        self.new_game_button = ttk.Button(control_frame, text="New Game", 
                                         command=self.new_game, state="disabled")
        self.new_game_button.grid(row=0, column=4, padx=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Select AI level and train or load a model", 
                                     font=('Arial', 10))
        self.status_label.grid(row=1, column=0, columnspan=5, pady=(10, 0))
        
        # Game board frame
        board_frame = ttk.LabelFrame(main_frame, text="Game Board", padding="10")
        board_frame.grid(row=2, column=0, columnspan=3, pady=(0, 20))
        
        # Create the game board
        self.create_board(board_frame)
        
        # Initialize scores
        self.player_score = 0
        self.ai_score = 0
        
    def create_board(self, parent):
        """Create the Connect4 game board"""
        self.board_buttons = []
        self.board_canvases = []
        
        for row in range(6):
            button_row = []
            canvas_row = []
            for col in range(7):
                # Create canvas for the cell
                canvas = tk.Canvas(parent, width=60, height=60, 
                                 bg=self.colors['empty'], 
                                 highlightthickness=1, 
                                 highlightbackground=self.colors['grid'])
                canvas.grid(row=row, column=col, padx=2, pady=2)
                
                # Bind click event
                canvas.bind('<Button-1>', lambda e, c=col: self.on_cell_click(c))
                
                canvas_row.append(canvas)
            self.board_canvases.append(canvas_row)
    
    def on_cell_click(self, col):
        """Handle cell click events"""
        if not self.game_active or not self.player_turn:
            return
            
        # Check if the column is valid
        state = self.env._get_observation()
        valid_actions = [i for i, valid in enumerate(state['action_mask']) if valid]
        
        if col not in valid_actions:
            messagebox.showwarning("Invalid Move", "This column is full!")
            return
        
        # Make player move
        self.make_move(col, is_player=True)
        
        # Check if game is over
        if self.check_game_over():
            return
        
        # AI turn
        self.player_turn = False
        self.root.update()
        
        # AI move with longer delay for better UX
        self.root.after(1500, self.ai_move)  # 1.5 second delay for visual effect
    
    def make_move(self, col, is_player=True):
        """Make a move on the board"""
        # Update environment
        time_step = self.env.step(col)
        
        # Update visual board
        self.update_board_display()
        
        # Check if game is over after this move
        if time_step.is_last:
            self.game_active = False
            # Check for win immediately
            if self.env._check_win(1):  # Player won
                self.player_score += 1
                messagebox.showinfo("🎉 Victory!", "Congratulations! You won!")
            elif self.env._check_win(-1):  # AI won
                self.ai_score += 1
                messagebox.showinfo("🤖 Defeat!", "The AI won this round!")
            elif np.all(self.env._board != 0):  # Draw
                messagebox.showinfo("🤝 Draw!", "It's a draw!")
            return
        
        # Switch turns manually since auto_opponent is False
        self.env._current_player = -self.env._current_player
        
        # Update turn indicator
        if is_player:
            # Player just moved, now it's AI's turn
            self.player_turn = False
        else:
            # AI just moved, now it's player's turn
            self.player_turn = True
    
    def ai_move(self):
        """Make AI move"""
        if not self.game_active or self.player_turn:
            return
            
        state = self.env._get_observation()
        valid_actions = [i for i, valid in enumerate(state['action_mask']) if valid]
        
        if not valid_actions:
            self.check_game_over()
            return
        
        # Get AI action
        if self.agent is None:
            # Random move if no agent loaded
            action = np.random.choice(valid_actions)
        else:
            # Get best action from agent
            board = np.array([state['board']])
            action_mask = np.array([state['action_mask']])
            q_values = self.agent.q_network.predict([board, action_mask], verbose=0)[0]
            
            valid_q_values = [(i, q_values[i]) for i in valid_actions]
            action = max(valid_q_values, key=lambda x: x[1])[0]
        
        # Make AI move
        self.make_move(action, is_player=False)
        
        # Check if game is over after AI move
        if self.game_active:
            self.check_game_over()
    
    def update_board_display(self):
        """Update the visual board display"""
        board = self.env._board
        
        for row in range(6):
            for col in range(7):
                canvas = self.board_canvases[row][col]
                canvas.delete("all")  # Clear canvas
                
                if board[row, col] == 0:
                    # Empty cell
                    canvas.configure(bg=self.colors['empty'])
                elif board[row, col] == 1:
                    # Player piece (red)
                    canvas.configure(bg=self.colors['player'])
                    canvas.create_oval(10, 10, 50, 50, fill=self.colors['player'], 
                                     outline=self.colors['grid'], width=2)
                else:
                    # AI piece (yellow)
                    canvas.configure(bg=self.colors['ai'])
                    canvas.create_oval(10, 10, 50, 50, fill=self.colors['ai'], 
                                     outline=self.colors['grid'], width=2)
    
    def check_game_over(self):
        """Check if the game is over and handle result"""
        if not self.game_active:
            return True
            
        # Check for win conditions using the environment's board
        if self.env._check_win(1):  # Player won (player is 1)
            self.game_active = False
            self.player_score += 1
            messagebox.showinfo("🎉 Victory!", "Congratulations! You won!")
            return True
            
        elif self.env._check_win(-1):  # AI won (AI is -1)
            self.game_active = False
            self.ai_score += 1
            messagebox.showinfo("🤖 Defeat!", "The AI won this round!")
            return True
            
        elif np.all(self.env._board != 0):  # Draw
            self.game_active = False
            messagebox.showinfo("🤝 Draw!", "It's a draw!")
            return True
            
        return False
    
    def train_agent(self):
        """Train the AI agent"""
        level = self.level_var.get()
        
        # Define training episodes based on level
        episodes_map = {
            "beginner": 100,
            "intermediate": 500,
            "expert": 1000
        }
        
        episodes = episodes_map[level]
        
        # Update status
        self.status_label.config(text=f"Training AI to {level} level ({episodes} episodes)...")
        self.train_button.config(state="disabled")
        self.root.update()
        
        try:
            # Create and train agent
            self.agent = SimpleConnect4Agent(self.env)
            self.agent.train(episodes=episodes)
            
            # Save the trained model
            model_path = f"trained_agent_{level}"
            self.agent.save_model(model_path)
            
            self.current_level = level
            self.status_label.config(text=f"AI trained to {level} level! Model saved.")
            self.new_game_button.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Training Error", f"Error during training: {str(e)}")
            self.status_label.config(text="Training failed. Please try again.")
        finally:
            self.train_button.config(state="normal")
    
    def load_agent(self):
        """Load a trained agent"""
        level = self.level_var.get()
        model_path = f"trained_agent_{level}"
        
        if not os.path.exists(model_path):
            messagebox.showerror("Load Error", 
                               f"No trained model found for {level} level. Please train first.")
            return
        
        try:
            # Create agent and load model
            self.agent = SimpleConnect4Agent(self.env)
            
            # Import the custom loss function
            from agent import mse_loss, ActionMaskLayer
            
            # Define custom objects for loading
            custom_objects = {
                'mse_loss': mse_loss,
                'ActionMaskLayer': ActionMaskLayer
            }
            
            # Try to load .keras format first, then fall back to .h5
            keras_path = os.path.join(model_path, 'q_network.keras')
            h5_path = os.path.join(model_path, 'q_network.h5')
            
            if os.path.exists(keras_path):
                self.agent.q_network = tf.keras.models.load_model(keras_path, custom_objects=custom_objects)
                self.agent.target_network = tf.keras.models.load_model(keras_path, custom_objects=custom_objects)
            elif os.path.exists(h5_path):
                self.agent.q_network = tf.keras.models.load_model(h5_path, custom_objects=custom_objects)
                self.agent.target_network = tf.keras.models.load_model(h5_path, custom_objects=custom_objects)
            else:
                messagebox.showerror("Load Error", 
                                   f"No model file found in {model_path}")
                return
            
            # Load training parameters
            params_path = os.path.join(model_path, 'params.npy')
            if os.path.exists(params_path):
                params = np.load(params_path, allow_pickle=True).item()
                self.agent.epsilon = params['epsilon']
                self.agent.learning_rate = params['learning_rate']
                self.agent.epsilon_decay = params['epsilon_decay']
                self.agent.epsilon_min = params['epsilon_min']
            
            self.current_level = level
            self.status_label.config(text=f"Loaded {level} level AI model!")
            self.new_game_button.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Error loading model: {str(e)}")
            self.status_label.config(text="Failed to load model.")
    
    def new_game(self):
        """Start a new game"""
        # Reset environment
        self.env = SimpleConnect4Environment(auto_opponent=False)
        
        # Reset game state
        self.game_active = True
        self.player_turn = True
        
        # Update display
        self.update_board_display()
        
        # Update status
        if self.agent:
            self.status_label.config(text=f"🎲 New game started! Playing against {self.current_level} AI.")
        else:
            self.status_label.config(text="🎲 New game started! Playing against random AI.")

def main():
    root = tk.Tk()
    app = Connect4GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 