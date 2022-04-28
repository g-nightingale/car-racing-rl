
import os
import numpy as np
import pickle
from collections import deque
import time
import random
import copy
import logging
import keras.backend as K
from IPython.display import clear_output
from utils import *


class DQNAgentTrainer:
    """A class to train a DQN agent.

    Attributes:
        img_len (int): Length and width of image (images must be sqaure).
        frame_stack_num (int): Number of frames to be stacked in grayscale image.
        frame_interval (int): Gap between stacked frames.
        img_dim (tuple): Image dimensions in height x width x depth.
        number_of_episodes (int): Number of episodes to train agent.
        epsilon (float): Starting epsilon value → probability of selecting a an action at random.
        epsilon_start (float): Starting epsilon value.
        epsilon_decay (bool): Use epsilon decay or not.
        epsilon_decay_rate (float): Epsilon decay rate - if 0.95, epsilon will be decayed to 5% of original value.
        epsilon_decay_steps (int): Number of steps over which to decay epsilon.
        max_replay_memory_size (int): Maximum number of observations in the replay memory.
        min_replay_memory_size (int): Minimum number of observations in the replay memory before model training starts.
        random_action_steps (int): Number of steps to take random action instead of using model.
        punishment_multiplier (float): Multiplier to increase negative rewards per consecutive negative reward in skip frames.
        max_consecutive_negative_rewards (int): Maximum number of consecutive negative rewards before ending episode.
        update_target_model_steps (int): Frequency at which to update the target model weights.
        target_model_hard_update (bool): Make hard update or soft update to target model weights. 
        max_steps_per_episode (int): Maximum number of steps before ending an episode.
        compound_action_frames (int): Number of frames to compund taking each action.
        save_model_frequency (int): Frequency at which the agent q-value model is saved.
        save_models (bool): Save agent q-value model or not.
        save_run_results (bool): Save results from train_agent() method.
        verbose_cnn (int): Frequency to print outputs from q-value training.
        verbose (bool): Print training details to terminal or not.
        plot_progress (bool): Plot current rewards or not.
        benchmark_rewards (list): Benchmark rewards to plot against - useful for comparing training progress.
        step_count (int): Count of steps elapsed.
        episode_rewards (list): List of rewards from the train_agent() method.
        
    """
    def __init__(self, img_len=56, frame_stack_num=4, frame_interval=1, number_of_episodes=1000, epsilon=1.0, epsilon_decay=False, epsilon_decay_rate=0.95,
                 epsilon_decay_steps=100000, max_replay_memory_size=100000, min_replay_memory_size=10000,                   
                 random_action_steps=2000, punishment_multiplier=4.0, max_consecutive_negative_rewards=50, update_target_model_steps=5000, 
                 target_model_hard_update=True, max_steps_per_episode=10000, compound_action_frames=4, save_model_frequency=10000, save_models=False, 
                 save_run_results=True, verbose_cnn=50, verbose=True, plot_progress=False, benchmark_rewards=None):

        self.img_len = img_len
        self.frame_stack_num = frame_stack_num
        self.frame_interval = frame_interval
        self.img_dim = (self.img_len, self.img_len, self.frame_stack_num)
        self.number_of_episodes = number_of_episodes
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_decay_steps = epsilon_decay_steps
        self.max_replay_memory_size = max_replay_memory_size
        self.min_replay_memory_size = min_replay_memory_size
        self.random_action_steps = random_action_steps
        self.punishment_multiplier = punishment_multiplier
        self.max_consecutive_negative_rewards = max_consecutive_negative_rewards
        self.update_target_model_steps = update_target_model_steps
        self.target_model_hard_update = target_model_hard_update
        self.max_steps_per_episode = max_steps_per_episode
        self.compound_action_frames = compound_action_frames
        self.save_model_frequency = save_model_frequency
        self.save_models = save_models
        self.save_run_results = save_run_results
        self.verbose_cnn = verbose_cnn
        self.verbose = verbose
        self.plot_progress = plot_progress
        self.benchmark_rewards = benchmark_rewards
        self.step_count = 0
        self.model_fits = 0
        self.grayscale_state_buffer = deque()
        self.episode_rewards = []
        self.path = None


    def train_agent(self, env, agent):
        """Train the DQN agent."""

        # Init variables
        episode_rewards =[]
        self.step_count = 0
        self.timestr = time.strftime('%Y%m%d-%H%M%S')
        self.path = f'./runs/{self.timestr}'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        log_filename = (f'{self.path}/log.log')
        logging.basicConfig(filename=log_filename, level=logging.INFO, force=True)

        # Output results
        if self.save_run_results:
            self.output_config_and_results(agent)
            
        # Create agent models
        agent.create_models(self.img_dim)

        for episode_number in range(self.number_of_episodes):

            # Reset the environment
            _ = env.reset()

            # First 50 frames are zooming into track so ignore for training!
            for _ in range(50):
                s_0, _, _, _ = env.step([0.0, 1.0, 0.0])

            # Generate episode
            episode_reward = self.generate_episode(s_0, env, agent, episode_number)
            episode_rewards.append(episode_reward)

            # Print and update log
            clear_output(wait=True)
            if self.verbose:
                print(f'Total step count: {self.step_count}')
                print(f'Epsilon: {self.epsilon}')
                print(f'Learning rate: {round(agent.model.optimizer.lr.numpy(), 5)}')
                print(f'Total reward for episode: {episode_reward}')
                print(f'Running average rewards: {np.mean(episode_rewards[-100:])} \n')

            if self.plot_progress:
                self.plot_rewards(episode_rewards)

            logging.info(f'Total step count: {self.step_count}')
            logging.info(f'Epsilon: {self.epsilon}')
            logging.info(f'Learning rate: {round(agent.model.optimizer.lr.numpy(), 5)}')
            logging.info(f'Total reward for episode: {episode_reward}')
            logging.info(f'Running average rewards: {np.mean(episode_rewards[-100:])} \n')

            # Save model
            if episode_number % self.save_model_frequency == 0 and episode_number > 0 and self.save_models is True:
                agent.model.save(f'{self.path}/model_{episode_number}.h5')

        env.close()

        # Save model
        if self.save_models is True:
            agent.model.save(f'{self.path}/model_final.h5')

        # Store rewards
        self.episode_rewards = episode_rewards

        # Output results
        if self.save_run_results:
            self.output_config_and_results(agent)


    def generate_episode(self, s_0, env, agent, episode_number=0):
        """Generate an episode of learning."""
        
        episode_reward = 0
        episode_step_count = 0
        consecutive_negative_rewards = 0

        # Get stacked state representation
        s_0 = image_processing(s_0, self.img_len)
        grayscale_state_buffer = [s_0] * self.frame_stack_num * self.frame_interval
        s_0_stacked = get_stacked_state(grayscale_state_buffer, self.frame_interval, self.img_dim)

        while True:

            # Increment counters
            episode_step_count += 1
            self.step_count += 1

            # Select action
            if self.step_count < self.random_action_steps or np.random.rand() < self.epsilon:
                a_0 = np.random.choice(np.arange(agent.num_actions), p=agent.action_probs)
            else:
                a_0 = np.argmax(agent.model.predict(s_0_stacked))

            # Action reward aggregates rewards over the skip frame loop
            action_reward = 0
            punishment_factor = 1.0
            for _ in range(self.compound_action_frames):
                # Take action and reshape new state
                s_1, reward, done, _ = env.step(agent.actions[a_0])

                # Append latest state and get new stacked representation
                s_1 = image_processing(s_1, self.img_len)
                grayscale_state_buffer.pop(0)
                grayscale_state_buffer.append(s_1)
                
                if reward < 0:
                    action_reward += reward * punishment_factor
                    punishment_factor *= self.punishment_multiplier
                else:
                    action_reward += reward
                    punishment_factor = 1.0
                episode_reward += reward
                if done:
                    break

            # Increment consecutive negative reward counter
            if action_reward < 0:
                consecutive_negative_rewards += 1 
            else: 
                consecutive_negative_rewards = 0
                
            # Append (s, a, r, s') to d
            s_1_stacked = get_stacked_state(grayscale_state_buffer, self.frame_interval, self.img_dim)
            agent.d.append((s_0_stacked, a_0, action_reward, s_1_stacked, done))

            # Start experience replay
            if len(agent.d) >= self.min_replay_memory_size:
                verbose = True if self.step_count % self.verbose_cnn == 0 else False
                agent.experience_replay(self.img_dim, self.step_count, verbose)
                self.model_fits += 1

            # Drop the oldest experience if the length of d exceeeds MAX_REPLAY_MEMORY_SIZE
            if len(agent.d) > self.max_replay_memory_size:
                agent.d.popleft()

            # Set target weights to model weights
            if self.step_count % self.update_target_model_steps == 0:
                agent.update_model_weights(self.target_model_hard_update)

            # Update epsilon
            if self.epsilon_decay is True and self.step_count > self.random_action_steps and self.step_count > self.min_replay_memory_size:
                self.decay_epsilon()

            # End the episode if following conditions are met
            if done or (episode_step_count > self.max_steps_per_episode) or (consecutive_negative_rewards > self.max_consecutive_negative_rewards):
                clear_output(wait=True)
                if self.verbose:
                    print(f'Episode: {episode_number}')
                    print(f'Ending episode: done {done}, consecutive negative rewards {consecutive_negative_rewards}')
                    print(f'Total model fits: {self.model_fits}')
                    print(f'Total steps in episode: {episode_step_count}')   

                logging.info(f'Episode: {episode_number}')
                logging.info(f'Ending episode: done {done}, consecutive negative rewards {consecutive_negative_rewards}')
                logging.info(f'Total model fits: {self.model_fits}')
                logging.info(f'Total steps in episode: {episode_step_count}')
                break

            # Current state ← new state
            s_0_stacked = s_1_stacked.copy()

        if self.verbose:
            print(f'Replay buffer size: {len(agent.d)}')
        logging.info(f'Replay buffer size: {len(agent.d)}')

        return episode_reward


    def decay_epsilon(self):
        """Decay the learning rate."""
        self.epsilon = self.epsilon_start * np.power((1 - self.epsilon_decay_rate), (min(self.step_count, self.epsilon_decay_steps) /self.epsilon_decay_steps))


    def plot_rewards(self, episode_rewards):
        """Plot episode rewards."""
        # Create new benchmark rewards list that is thesame length as episode rewards
        if self.benchmark_rewards is not None:
            benchmark_rewards = self.benchmark_rewards[:len(episode_rewards)]

        # Plot
        plt.figure(figsize=(8, 6))
        plt.title('DQN Agent')
        plt.plot(episode_rewards, label='Episode reward', alpha=0.5)
        plt.plot([np.mean(episode_rewards[::-1][i:i+100]) for i in range(len(episode_rewards))][::-1], label='Average reward (last 100 episodes)')
        if self.benchmark_rewards is not None:
            plt.plot([np.mean(benchmark_rewards[::-1][i:i+100]) for i in range(len(benchmark_rewards))][::-1], label='Benchmark average reward (last 100 episodes)')
        plt.xlabel('episode')
        plt.ylabel('average reward')
        plt.legend()
        plt.savefig(f'{self.path}/training_rewards.png')
        plt.show()
        

    def output_config_and_results(self, agent):
        """Outputs config and results to a pickle file."""

        config = {}
        # Trainer config
        config['img_len'] = self.img_len
        config['frame_stack_num'] = self.frame_stack_num
        config['img_dim'] = self.img_dim
        config['number_of_episodes'] = self.number_of_episodes 
        config['epsilon_start'] = self.epsilon_start
        config['epsilon'] = self.epsilon
        config['epsilon_decay'] = self.epsilon_decay
        config['epsilon_decay_rate'] = self.epsilon_decay_rate
        config['epsilon_decay_steps'] = self.epsilon_decay_steps
        config['max_replay_memory_size'] = self.max_replay_memory_size
        config['min_replay_memory_size'] = self.min_replay_memory_size 
        config['random_action_steps'] = self.random_action_steps
        config['punishment_multiplier'] = self.punishment_multiplier
        config['max_consecutive_negative_rewards'] = self.max_consecutive_negative_rewards
        config['update_target_model_steps'] = self.update_target_model_steps
        config['target_model_hard_update'] = self.target_model_hard_update
        config['max_steps_per_episode'] = self.max_steps_per_episode
        config['compound_action_frames'] = self.compound_action_frames
        config['save_model_frequency'] = self.save_model_frequency
        config['save_models'] = self.save_models
        config['save_run_results'] = self.save_run_results
        config['verbose_cnn'] = self.verbose_cnn
        config['verbose'] = self.verbose
        config['final_step_count'] = self.step_count
        config['model_fits'] = self.model_fits

        # Agent config
        config['actions'] = agent.actions
        config['num_actions'] = agent.num_actions
        config['action_probs'] = agent.action_probs   
        config['lr'] = agent.lr
        config['batch_size'] = agent.batch_size
        config['gamma'] = agent.gamma
        config['tau'] = agent.tau
        config['ddqn'] = agent.ddqn
        config['decay_lr'] = agent.decay_lr
        config['decay_rate'] = agent.decay_rate
        config['decay_steps'] = agent.decay_steps
        config['decay_step_start'] = agent.decay_step_start

        # Rewards
        config['episode_rewards'] = self.episode_rewards
        
        # Save
        with open(f'{self.path}/config_results.pickle', 'wb') as handle:
            pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

