import gym
import numpy as np
from gym.wrappers import Monitor
import glob
import io
import base64
from IPython.display import HTML
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from collections import deque
import time
import random
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, \
                        GlobalMaxPool2D, BatchNormalization, Dropout, Activation
from keras.backend import clear_session

from dqn_agent_trainer import *
from dqn_agent import *



if __name__ == "__main__":
    """Train the agent."""
    
    ACTIONS = np.array([[0.0, 1.0, 0.0],  
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.8],
                    [0.0, 0.0, 0.0]]   
                )
    ACTION_PROBS = np.array([0.5] + [(1 - 0.5)/4.0] * 4)

    # Create a DQN agent
    dqn_agent = DQNAgent(ACTIONS,
                        ACTION_PROBS,
                        lr=0.00025, 
                        batch_size=32, 
                        gamma=0.95)

    # Create the agent trainer
    agent_trainer = DQNAgentTrainer(img_len=56, 
                                    frame_stack_num=4, 
                                    number_of_episodes=1000, 
                                    epsilon=0.1, 
                                    epsilon_min=0.05, 
                                    epsilon_step_episodes=50.0,
                                    final_epsilon_episode=10000,
                                    max_replay_memory_size=100000, 
                                    min_replay_memory_size=1000, 
                                    random_action_frames=2000,
                                    max_consecutive_negative_rewards=50, 
                                    update_target_model_frames=5000, 
                                    max_frames_per_episode=10000,
                                    skip_frames=4, 
                                    save_training_frequency=10000, 
                                    save_models=False, 
                                    save_run_results=True,
                                    verbose_cnn=50, 
                                    verbose=True)

    # Define environment and train agent
    env = gym.make("CarRacing-v0")
    agent_trainer.train_agent(env, dqn_agent)
