import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics

class Connect4QNetwork(q_network.QNetwork):

    def __init__(self, input_tensor_spec, action_spec, name='Connect4QNetwork'):
        super(Connect4QNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            action_spec=action_spec,
            name=name
        )
        
        # Define the network architecture
        self._flatten = tf.keras.layers.Flatten()
        self._dense1 = tf.keras.layers.Dense(128, activation='relu')
        self._dense2 = tf.keras.layers.Dense(128, activation='relu')
        self._dense3 = tf.keras.layers.Dense(64, activation='relu')
        self._output_layer = tf.keras.layers.Dense(action_spec.maximum + 1)
    
    def call(self, observation, step_type=None, network_state=(), training=False):
        # Extract board and action mask from observation
        board = observation['board']
        action_mask = observation['action_mask']
        
        # Flatten the board for processing
        board_flat = self._flatten(board)
        
        # Pass through dense layers
        x = self._dense1(board_flat)
        x = self._dense2(x)
        x = self._dense3(x)
        q_values = self._output_layer(x)
        
        # Apply action masking - set invalid actions to very negative values
        # Convert boolean mask to float and invert it (1 for invalid, 0 for valid)
        mask = tf.cast(tf.logical_not(action_mask), tf.float32)
        masked_q_values = q_values - mask * 1e9
        
        return masked_q_values, network_state

class Connect4Agent:
    
    def __init__(self, train_env, eval_env, learning_rate=1e-3, replay_buffer_capacity=100000):
        self.train_env = train_env
        self.eval_env = eval_env
        
        # Create the Q-Network
        self.q_net = Connect4QNetwork(
            input_tensor_spec=train_env.observation_spec(),
            action_spec=train_env.action_spec()
        )
        
        # Create the agent
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=self.q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=tf.Variable(0)
        )
        self.agent.initialize()
        
        # Create replay buffer. Still a bit confused by this.
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=replay_buffer_capacity
        )
        
        self.collect_policy = self.agent.collect_policy # Collection policy uses ε-greedy
        self.eval_policy = self.agent.policy # Evaluation policy using pure greedy
        
        self.collect_driver = dynamic_step_driver.DynamicStepDriver(
            train_env,
            self.collect_policy,
            observers=[self.replay_buffer.add_batch],
            num_steps=1
        )
        
        # Initialize metrics
        self.train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
        ]
        
        self.eval_metrics = [
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
        ]
    
    def collect_initial_data(self, num_steps=1000):
        """Collect initial random data to populate replay buffer."""
        print(f"Collecting {num_steps} initial steps...")
        
        # Use random policy for initial data collection
        random_policy = random_tf_policy.RandomTFPolicy(
            self.train_env.time_step_spec(),
            self.train_env.action_spec()
        )
        
        initial_driver = dynamic_step_driver.DynamicStepDriver(
            self.train_env,
            random_policy,
            observers=[self.replay_buffer.add_batch],
            num_steps=num_steps
        )
        
        initial_driver.run()
        print(f"Replay buffer now contains {self.replay_buffer.num_frames()} frames")
    
    def train_step(self, batch_size=64):
        """Perform one training step."""
        # Sample a batch from replay buffer
        experience, unused_info = next(
            iter(self.replay_buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=batch_size,
                num_steps=2
            ).prefetch(3))
        )
        
        # Train the agent
        train_loss = self.agent.train(experience).loss
        return train_loss
    
    def evaluate(self, num_episodes=10):
        """Evaluate the agent's performance."""
        print(f"Evaluating for {num_episodes} episodes...")
        
        for metric in self.eval_metrics:
            metric.reset()
        
        for _ in range(num_episodes):
            time_step = self.eval_env.reset()
            episode_return = 0
            
            while not time_step.is_last():
                action_step = self.eval_policy.action(time_step)
                time_step = self.eval_env.step(action_step.action)
                episode_return += time_step.reward
            
            # Update metrics
            for metric in self.eval_metrics:
                metric(trajectory.Trajectory(
                    step_type=time_step.step_type,
                    observation=time_step.observation,
                    action=tf.constant([0]),  # Dummy action for metrics
                    policy_info=(),
                    next_step_type=time_step.step_type,
                    reward=episode_return,
                    discount=time_step.discount
                ))
        
        # Print evaluation results
        results = {}
        for metric in self.eval_metrics:
            results[metric.name] = metric.result().numpy()
            print(f"{metric.name}: {results[metric.name]:.2f}")
        
        return results
    
    def train(self, num_iterations=10000, collect_steps_per_iteration=1, 
              batch_size=64, eval_interval=1000, log_interval=200):
        """Train the agent."""
        print("Starting training...")
        
        # Collect initial data
        self.collect_initial_data()
        
        for iteration in range(num_iterations):
            # Collect data
            for _ in range(collect_steps_per_iteration):
                self.collect_driver.run()
            
            # Train
            train_loss = self.train_step(batch_size)
            
            # Update metrics
            for metric in self.train_metrics:
                metric.reset()
            
            # Log progress
            if iteration % log_interval == 0:
                print(f"Iteration {iteration}: Loss = {train_loss:.4f}")
                print(f"Replay buffer size: {self.replay_buffer.num_frames()}")
            
            # Evaluate
            if iteration % eval_interval == 0:
                self.evaluate()
    
    def save_model(self, filepath):
        """Save the trained model."""
        tf.saved_model.save(self.agent.policy, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        self.eval_policy = tf.saved_model.load(filepath)
        print(f"Model loaded from {filepath}")

def main():
    from environment import Connect4Environment
    
    train_py_env = Connect4Environment()
    eval_py_env = Connect4Environment()
    
    # Convert py_environments to TF environments using TFPyEnvironment wrapper 
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    
    # Create and train the agent
    agent = Connect4Agent(train_env, eval_env, learning_rate=1e-3)
    agent.train(
        num_iterations=20000,
        collect_steps_per_iteration=1,
        batch_size=64,
        eval_interval=1000,
        log_interval=200
    )
    
    agent.save_model("connect4_model")
    
    print("\nFinal Evaluation:")
    agent.evaluate(num_episodes=50)

if __name__ == "__main__":
    main()