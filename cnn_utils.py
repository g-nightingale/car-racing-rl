import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, \
                        GlobalMaxPool2D, BatchNormalization, Dropout, Activation
from keras.backend import clear_session


def create_activation_model(model):
    """Creates an activation model from a given CNN model."""
    layer_outputs = []
    layer_names = [] 
    for layer in model.layers:
        if isinstance(layer, (Conv2D, MaxPooling2D)):
            layer_outputs.append(layer.output)
            layer_names.append(layer.name)
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    return activation_model


def display_cnn_feature_maps_and_filters(model, activation_model, digit, total_width=10, 
                                         cmap='viridis'):
    """Display the feature maps and filters for a CNN.
       https://stackoverflow.com/questions/28933233/embedding-multiple-gridspec-layouts-on-a-single-matplotlib-figure """

    # Get activations for a given digit
    activations = activation_model.predict(digit)
    
    # Get the convolutional layers from the original model
    layers = [layer for layer in model.layers if 'conv' in layer.name]

    for i, layer in enumerate(layers):
        # Retrieve weights from the hidden layer
        filters, biases = layer.get_weights()
        n_filters = filters.shape[3]

        activation_layer = activations[i] 
        n_featuremaps = activation_layer.shape[3]

        # Normalize filter values to 0-1
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)

        # Calculate 
        cols_n = 8
        if n_filters % np.sqrt(n_filters) == 0:
            cols_n = int(np.sqrt(n_filters))
        rows_n = int(n_filters/cols_n)
        width = (total_width - 1)/2
        height = width/cols_n * rows_n

        # Set-up gridspec
        plt.figure(figsize=(width*2 + 1, height))
        outer_grid = gridspec.GridSpec(1, 3, width_ratios=[0.45, 0.1, 0.45], 
                                       wspace=0.00, hspace=0.00)

        left_cell = outer_grid[0, 0]
        right_cell = outer_grid[0, 2]

        gs1 = gridspec.GridSpecFromSubplotSpec(rows_n, cols_n, left_cell, wspace=0.00, hspace=0.00)
        gs2 = gridspec.GridSpecFromSubplotSpec(rows_n, cols_n, right_cell, wspace=0.00, hspace=0.00)

        # Plot featuremaps
        print(layer.name)
        
        for i in range(n_featuremaps):
            ax = plt.subplot(gs1[i])
            ax.matshow(activation_layer[0, :, :, i], cmap=cmap)
            ax.xaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_major_locator(ticker.NullLocator())
            ax.set_aspect('equal')

        # Plot filters
        for i in range(n_filters):
            f = filters[:, :, :, i]
            ax = plt.subplot(gs2[i])
            ax.matshow(f[:, :, 0], cmap=cmap)
            ax.xaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_major_locator(ticker.NullLocator())
            ax.set_aspect('equal')
            
        plt.show()