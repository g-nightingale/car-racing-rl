
import numpy as np
import pickle
from collections import deque
import time
import random
import copy
import logging
from IPython.display import clear_output
from utils import *


class DQNAgentTrainer:
    """A class to train a DQN agent.

    Attributes:
        img_len (int): Length and width of image (images must be sqaure).
        frame_stack_num (int): Number of frames to be stacked in grayscale image.
        img_dim (tuple): Image dimensions in height x width x depth.
        number_of_episodes (int): Number of episodes to train agent.
        epsilon (float): Starting epsilon value → probability of selecting a an action at random.
        epsilon_min (float): Mimum epsilon value.
        epsilon_step_episodes (float): Number of episodes to step from starting epsilon to epsilon_min.
        max_replay_memory_size (int): Maximum number of observations in the replay memory.
        min_replay_memory_size (int): Minimum number of observations in the replay memory before model training starts.
        random_action_steps (int): Number of steps to take random action instead of using model.
        max_consecutive_negative_rewards (int): Maximum number of consecutive negative rewards before ending episode.
        update_target_model_steps (int): Frequency at which to update the target model weights.
        max_steps_per_episode (int): Maximum number of steps before ending an episode.
        skip_frames (int): Number of frames to skip after taking each action.
        save_training_frequency (int): Frequency at which the agent q-value model is saved.
        save_models (bool): Save agent q-value model or not.
        save_run_results (bool): Save results from train_agent() method.
        verbose_cnn (int): Frequency to print outputs from q-value training.
        verbose (bool): Print training details to terminal or not.
        step_count (int): Count of steps elapsed.
        episode_rewards (list): List of rewards from the train_agent() method.
        
    """
    def __init__(self, img_len=56, frame_stack_num=4, number_of_episodes=1000, epsilon=1.0, epsilon_min=0.05, epsilon_step_episodes=100.0,
                 final_epsilon_episode=500, max_replay_memory_size=100000, min_replay_memory_size=10000, random_action_steps=2000,
                 max_consecutive_negative_rewards=50, update_target_model_steps=5000, max_steps_per_episode=10000,
                 skip_frames=4, save_training_frequency=10000, save_models=False, save_run_results=True, verbose_cnn=50, verbose=True):

        self.img_len = img_len
        self.frame_stack_num = frame_stack_num
        self.img_dim = (self.img_len, self.img_len, self.frame_stack_num)
        self.number_of_episodes = number_of_episodes
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_step_episodes = epsilon_step_episodes
        self.epsilon_step = (self.epsilon - self.epsilon_min)/self.epsilon_step_episodes
        self.final_epsilon_episode = final_epsilon_episode
        self.max_replay_memory_size = max_replay_memory_size
        self.min_replay_memory_size = min_replay_memory_size
        self.random_action_steps = random_action_steps
        self.max_consecutive_negative_rewards = max_consecutive_negative_rewards
        self.update_target_model_steps = update_target_model_steps
        self.max_steps_per_episode = max_steps_per_episode
        self.skip_frames = skip_frames
        self.save_training_frequency = save_training_frequency
        self.save_models = save_models
        self.save_run_results = save_run_results
        self.verbose_cnn = verbose_cnn
        self.verbose = verbose
        self.step_count = 0
        self.grayscale_state_buffer = deque()
        self.episode_rewards = []


    def train_agent(self, env, agent):
        """Train the DQN agent."""

        # Init variables
        episode_rewards =[]
        self.step_count = 0
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        log_filename = (f'./logs/log_{self.timestr}.log')
        logging.basicConfig(filename=log_filename, level=logging.INFO, force=True)

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
                print(f'Total reward for episode: {episode_reward}')
                print(f'Running average rewards: {np.mean(episode_rewards[-100:])} \n')

            logging.info(f'Total step count: {self.step_count}')
            logging.info(f'Epsilon: {self.epsilon}')
            logging.info(f'Total reward for episode: {episode_reward}')
            logging.info(f'Running average rewards: {np.mean(episode_rewards[-100:])} \n')

            # Start decaying epsilon when policy is used to select actions
            if episode_number > self.final_epsilon_episode:
                self.epsilon = 0.0
            elif self.step_count > self.random_action_steps and self.step_count > self.min_replay_memory_size:
                self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)

        env.close()

        # Save model
        if self.save_models is True:
            agent.model.save_model(f'./saved_models/dqn_final.h5')

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
        grayscale_state_buffer = deque([s_0] * self.frame_stack_num)
        s_0_stacked = get_stacked_state(grayscale_state_buffer, self.img_dim)

        while True:

            # Increment counters
            episode_step_count += 1
            self.step_count += 1

            # Select action
            if self.step_count < self.random_action_steps or np.random.rand() < self.epsilon:
                a_0 = np.random.choice(np.arange(agent.num_actions), p=agent.action_probs)
                #a_0 = random.choice(np.arange(num_actions))
            else:
                a_0 = np.argmax(agent.model.predict(s_0_stacked))

            # Action reward aggregates rewards over the skip frame loop
            action_reward = 0
            for _ in range(self.skip_frames):
                # Take action and reshape new state
                s_1, reward, done, _ = env.step(agent.actions[a_0])
                
                action_reward += reward
                episode_reward += reward
                if done:
                    break

            # Increment consecutive negative reward counter
            if action_reward < 0:
                consecutive_negative_rewards += 1 
            else: 
                consecutive_negative_rewards = 0

            # Append latest state and get new stacked representation
            s_1 = image_processing(s_1, self.img_len)
            grayscale_state_buffer.append(s_1)
            s_1_stacked = get_stacked_state(grayscale_state_buffer, self.img_dim)

            # Append (s, a, r, s') to d
            agent.d.append((s_0_stacked, a_0, action_reward, s_1_stacked, done))

            # Start experience replay
            if len(agent.d) >= self.min_replay_memory_size:
                verbose = True if self.step_count % self.verbose_cnn == 0 else False
                agent.experience_replay(self.img_dim, verbose)

            # Drop the oldest experience if the length of d exceeeds MAX_REPLAY_MEMORY_SIZE
            if len(agent.d) > self.max_replay_memory_size:
                agent.d.popleft()

            # Set target weights to model weights
            if self.step_count % self.update_target_model_steps == 0:
                agent.update_model_weights()

            # Save model
            if self.step_count % self.save_training_frequency == 0 and self.save_models is True:
                agent.model.save_model(f'./saved_models/dqn_trial_{step_count}.h5')

            # End the episode if following conditions are met
            if done or (episode_step_count > self.max_steps_per_episode) or (consecutive_negative_rewards > self.max_consecutive_negative_rewards):
                clear_output(wait=True)
                if self.verbose:
                    print(f'Episode: {episode_number}')
                    print(f"Ending episode: done {done}, consecutive negative rewards {consecutive_negative_rewards}")
                    print(f"Total steps in episode: {episode_step_count}")   

                logging.info(f'Episode: {episode_number}')
                logging.info(f"Ending episode: done {done}, consecutive negative rewards {consecutive_negative_rewards}")
                logging.info(f"Total steps in episode: {episode_step_count}")
                break

            # Current state ← new state
            s_0_stacked = s_1_stacked.copy()

        if self.verbose:
            print(f'Replay buffer size: {len(agent.d)}')
        logging.info(f'Replay buffer size: {len(agent.d)}')

        return episode_reward


    def output_config_and_results(self, agent):
        """Outputs config and results to a pickle file."""

        config = {}
        # Trainer config
        config['img_len'] = self.img_len
        config['frame_stack_num'] = self.frame_stack_num
        config['img_dim'] = self.img_dim
        config['number_of_episodes'] = self.number_of_episodes 
        config['epsilon'] = self.epsilon
        config['epsilon_min'] = self.epsilon_min
        config['epsilon_step_episodes'] = self.epsilon_step_episodes 
        config['epsilon_step'] = self.epsilon_step 
        config['final_epsilon_episode'] = self.final_epsilon_episode
        config['max_replay_memory_size'] = self.max_replay_memory_size
        config['min_replay_memory_size'] = self.min_replay_memory_size 
        config['random_action_steps'] = self.random_action_steps
        config['max_consecutive_negative_rewards'] = self.max_consecutive_negative_rewards
        config['update_target_model_steps'] = self.update_target_model_steps
        config['max_steps_per_episode'] = self.max_steps_per_episode
        config['skip_frames'] = self.skip_frames
        config['save_training_frequency'] = self.save_training_frequency
        config['save_models'] = self.save_models
        config['save_run_results'] = self.save_run_results
        config['verbose_cnn'] = self.verbose_cnn
        config['verbose'] = self.verbose
        
        # Agent config
        #config['d'] = agent.d
        # agent_copy = copy.deepcopy(agent)
        # agent_copy.d = deque()
        # config['agent'] = agent_copy
        config['actions'] = agent.actions
        config['num_actions'] = agent.num_actions
        config['action_probs'] = agent.action_probs   
        config['lr'] = agent.lr
        config['batch_size'] = agent.batch_size
        config['gamma'] = agent.gamma
        config['model'] = agent.model
        config['target_model'] = agent.target_model

        # Rewards
        config['episode_rewards'] = self.episode_rewards

        # Save
        with open(f'./runs/dqn_results_{self.timestr}.pickle', 'wb') as handle:
            pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

