import gym
import numpy as np
from gym.wrappers import Monitor
from IPython.display import HTML
from pyvirtualdisplay import Display
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
                        GlobalMaxPool2D, BatchNormalization, Dropout, \
                        Activation, Rescaling
from keras.regularizers import l2
from keras.backend import clear_session
from dqn_agent import *
from dqn_agent_trainer import *
from utils import *
from cnn_utils import *

display = Display(visible=0, size=(1400, 900))
display.start()


# Define action set
actions = np.array([[0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.8],
                    [0.0, 0.0, 0.0]]   
                )

# Define action probabilities
action_probs = np.array([0.2] * 5)

# Define CNN model
def modified_cnn_model(img_dim, learning_rate, num_actions):
    """Returns a CNN model - modified architecture from DeepMind paper."""

    inputs = keras.Input(shape=img_dim)
    x = Rescaling(1./255) (inputs)
    x = Conv2D(filters=32, kernel_size=8, strides=4, kernel_regularizer=l2(1e-5)) (x)
    x = Activation('relu') (x)

    x = Conv2D(filters=64, kernel_size=4, strides=2, kernel_regularizer=l2(1e-5)) (x)
    x = Activation('relu') (x)
  
    x = Conv2D(filters=64, kernel_size=3, strides=1, kernel_regularizer=l2(1e-5)) (x)
    x = Activation('relu') (x)

    x = Flatten() (x)
    x = Dense(512, kernel_regularizer=l2(1e-5)) (x)
    x = Activation('relu') (x)

    outputs = Dense(num_actions, activation='linear', kernel_regularizer=l2(1e-5)) (x)
    model = Model(inputs=inputs, outputs=outputs)

    opt = Adam(learning_rate=learning_rate, epsilon=1e-7, clipnorm=1.0)
    loss_function = keras.losses.Huber()
    model.compile(loss=loss_function, optimizer=opt, metrics=['mean_squared_error'])

    return model

# Create a DQN agent
dqn_agent = DQNAgent(actions,
                    action_probs,
                    lr=0.001, 
                    batch_size=32,
                    gamma=0.99,
                    decay_lr=True,
                    decay_rate=0.9,
                    decay_steps=200000,
                    decay_step_start=0,
                    user_defined_model_function=modified_cnn_model,
                    ddqn=True)

# Create the agent trainer
agent_trainer = DQNAgentTrainer(img_len=96, 
                                frame_stack_num=4, 
                                frame_interval=4,
                                number_of_episodes=1000, 
                                epsilon=0.05, 
                                epsilon_decay=True, 
                                epsilon_decay_rate=0.9,
                                epsilon_decay_steps=36000,
                                max_replay_memory_size=100000, 
                                min_replay_memory_size=1000, 
                                random_action_steps=5000,
                                max_consecutive_negative_rewards=150, 
                                update_target_model_steps=5000, 
                                compound_action_frames=4, 
                                save_model_frequency=200, 
                                save_models=True, 
                                save_run_results=True,
                                verbose_cnn=50, 
                                verbose=True)

# Train the agent
env = gym.make("CarRacing-v0", verbose=0)
agent_trainer.train_agent(env, dqn_agent)
