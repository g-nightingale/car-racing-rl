import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from collections import deque
from itertools import permutations
import time
import random
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, \
                        GlobalMaxPool2D, BatchNormalization, Dropout, Activation
from keras.backend import clear_session
from utils import *


class DQNAgent:
    """Implementation of a DQN agent.

    Creates a DQN agent.

    Attributes:
        d (deque): Experience replay buffer.
        actions (np.array): Array of actions the agent can take.
        num_actions (int): Number of actions the agent can take. 
        action_probs (list): Probabilities when choosing actions randomly.
        lr (float): Learning rate for CNN model.
        batch_size (int): Batch size for experience replay sampling.
        gamma (float): Discount rate applied in q-value calculation.
        tau (float): Tau value used for soft model updates.
        decay_lr (bool): Use learning rate decay or not.
        decay_rate (float): Rate to decay learning rate.
        decay_steps (int): Number of steps to decay learning rate.
        decay_step_start (int): Starting step to decay learning rate.
        model (keras model): Model for predicting action values.
        target model (keras model): Model for updating action values.
        user_defined_model_function (function): User defined function to create a CNN model.
        ddqn (bool): Use double q-network or not.
    """

    def __init__(self, actions, action_probs=None, lr=0.00025, batch_size=32, gamma=0.95, tau=0.001,
                 decay_lr=False, decay_rate=0.95, decay_steps=100000, decay_step_start=0,
                 user_defined_model_function=None, ddqn=False):
        """Initialise the agent."""

        self.d = deque()
        self.actions = actions
        self.num_actions = self.actions.shape[0]
        if action_probs is None:
            self.action_probs = np.ones(self.num_actions) / self.num_actions
        else:
            self.action_probs = action_probs
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.model = None
        self.target_model = None
        self.decay_lr = decay_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.decay_step_start = decay_step_start
        if user_defined_model_function is not None:
            self.user_defined_model_function = user_defined_model_function
        else:
            self.user_defined_model_function = None
        self.ddqn = ddqn


    def cnn_model(self, img_dim):
        """Returns a CNN model - same architecture as in the DeepMind paper."""

        model = Sequential()
        model.add(Conv2D(32, 8, activation='relu', strides=4, input_shape=img_dim))
        model.add(Conv2D(64, 4, activation='relu', strides=2))
        model.add(Conv2D(64, 3, activation='relu', strides=1))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.num_actions, activation="linear"))

        opt = Adam(learning_rate=self.lr, epsilon=1e-7, clipnorm=1.0)
        loss_function = keras.losses.Huber()
        model.compile(loss=loss_function, optimizer=opt, metrics=['mean_squared_error'])

        return model


    def create_models(self, img_dim):
        """Create the models for the agent."""
        
        if self.user_defined_model_function is None:
            self.model = self.cnn_model(img_dim)
            self.target_model = self.cnn_model(img_dim)
        else:
            self.model = self.user_defined_model_function(img_dim, self.lr, self.num_actions)
            self.target_model = self.user_defined_model_function(img_dim, self.lr, self.num_actions)


    def experience_replay(self, img_dim, steps, verbose=False):
        """Experience replay step."""

        # Create random batch from replay buffer
        batch = random.sample(self.d, self.batch_size)

        # Get states
        s_0 = np.array([np.reshape(x[0], img_dim) for x in batch])
        s_1 = np.array([np.reshape(x[3], img_dim) for x in batch])

        # Get q values from models
        q_0 = self.model.predict(s_0)
        q_1 = self.target_model.predict(s_1)
        if self.ddqn:
            q_1_ddqn = self.model.predict(s_1)

        # Loop through batch and update q-values
        for i, (_, a, r, _, done) in enumerate(batch):
            if done:
                q_new = r
            else: 
                if self.ddqn:
                    q_new = r + self.gamma * q_1[i][np.argmax(q_1_ddqn[i])]
                else:
                    q_new = r + self.gamma * np.max(q_1[i])

            # Update the q-value
            q_0[i][a] = q_new 

        # Fit model
        self.model.fit(s_0, q_0, epochs=1, batch_size=self.batch_size, verbose=verbose)

        # Decay the learning rate
        if self.decay_lr and steps > self.decay_step_start:
            self.decay_learning_rate(steps)


    def update_model_weights(self, hard_update=True):
        """Update target model weights to match model weights."""
        if hard_update:
            self.target_model.set_weights(self.model.get_weights())
        else:
            new_weights = [(1 - self.tau) * tw  + self.tau * mw for tw, mw in zip(self.target_model.get_weights(), self.model.get_weights())]
            self.target_model.set_weights(new_weights)


    def save_model(self, path='./saved_models/dqn_final.h5'):
        """Save the model."""
        self.model.save(path)


    def decay_learning_rate(self, steps):
        """Decay the learning rate."""
        self.model.optimizer.lr = self.lr * np.power((1 - self.decay_rate), (min(steps, self.decay_steps)/self.decay_steps))


    def act(self, env, render=False):
        """Take action using learned behaviours."""
        s_0 = env.reset()
        episode_rewards = 0
        s_0 = image_processing(s_0)
        grayscale_state_buffer = deque([s_0]*9)
        s_0_stacked = get_stacked_state_live(grayscale_state_buffer)

        while True:
            if render:
                env.render()
            a_0 = np.argmax(self.model.predict(s_0_stacked))
            s_1, reward, done, _ = env.step(self.actions[a_0])
            episode_rewards += reward

            if done:
                break

            # Append latest state and get new stacked representation
            s_1 = image_processing(s_1)
            grayscale_state_buffer.append(s_1)
            s_0_stacked = get_stacked_state_live(grayscale_state_buffer)

        env.close()

        return episode_rewards