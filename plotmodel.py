#!/usr/bin/env python

import json
import numpy as np
import matplotlib.pyplot as plt
import model


def plot_model(w, b):
    """Show the weights as a 64x64 plot."""
    # Arrange the weights into a 64x64 RGB matrix such that the weights
    # in w[i, j] represents the weights of R, G and B components of the
    # corresponding pixel (i, j) in the input image.
    w = np.array(w).reshape(64, 64, 3)

    # Normalize the weights to real numbers between 0 and 1.
    w_norm = (w - np.min(w)) / (np.max(w) - np.min(w))
    w_r_norm = select_and_normalize_weights(w, 0)
    w_g_norm = select_and_normalize_weights(w, 1)
    w_b_norm = select_and_normalize_weights(w, 2)

    # Plot the weights.
    plt.imsave('plots/w.png', w_norm)
    plt.imsave('plots/wr.png', w_r_norm)
    plt.imsave('plots/wg.png', w_g_norm)
    plt.imsave('plots/wb.png', w_b_norm)


def plot_activations(w, b):
    """Show the activations as a 64x64 plot."""
    w = np.array(w)
    x = np.identity(w.shape[0])
    a = model.sigmoid(np.dot(w.T, x) + b)
    a = a.reshape(64, 64, 3)

    # Normalize the weights to real numbers between 0 and 1.
    a_norm = (a - np.min(a)) / (np.max(a) - np.min(a))
    a_r_norm = select_and_normalize_activations(a, 0)
    a_g_norm = select_and_normalize_activations(a, 1)
    a_b_norm = select_and_normalize_activations(a, 2)

    # Plot the weights.
    plt.imsave('plots/a.png', a_norm)
    plt.imsave('plots/ar.png', a_r_norm)
    plt.imsave('plots/ag.png', a_g_norm)
    plt.imsave('plots/ab.png', a_b_norm)


def select_and_normalize_weights(w, i):
    """Select and normalize the colors in a specific channel.

    The negative and positive weights are normalized separately to
    values between 0 and 1.

    The positive weights of a selected channel is represented with the
    corresponding color. For example, if the red channel is selected
    then a weight 1 is displayed as red color of full intensity.

    The negative weights of a selected channel is represented with it
    complementary secondary color. For example, if the red channel is
    selected then a weight 1 is displayed as cyan of full intensity.

    w (numpy.ndarray): Weights
    i (int): Channel number (0 for red, 1 for green, 2 for blue)
    """
    w = w.copy()

    # Let i be the selected channel and j be the other channels.
    if i == 0:
        j, k = 1, 2
    elif i == 1:
        j, k = 0, 2
    elif i == 2:
        j, k = 0, 1

    # Set channels j and k to 0 thereby effectively selecting channel i.
    w[:,:,j] = w[:,:,k] = 0

    # Get the max and min of channel i. These are going to be
    # denominators while normalizing.
    w_max = np.max(w)
    w_min = np.min(w)

    # Normalize all the weights.
    for r in range(64):
        for c in range(64):
            rgb = w[r][c]
            if rgb[i] > 0:
                rgb[i] = rgb[i] / w_max
            else:
                rgb[j] = rgb[k] = rgb[i] / w_min
                rgb[i] = 0

    # Return normalized weights.
    return w


def select_and_normalize_activations(a, i):
    """Select and normalize the colors in a specific channel.
    
    The activation values are normalized to values between 0 and 1.
    
    w (numpy.ndarray): Activations
    i (int): Channel number (0 for red, 1 for green, 2 for blue)
    """
    a = a.copy()

    # Let i be the selected channel and j be the other channels.
    if i == 0:
        j, k = 1, 2
    elif i == 1:
        j, k = 0, 2
    elif i == 2:
        j, k = 0, 1

    # Set channels j and k to 0 thereby effectively selecting channel i.
    a[:,:,j] = a[:,:,k] = 0

    # Find the maximum and minimum value in the selected channel.
    channel = a[:,:,i]
    a_max = np.max(channel)
    a_min = np.min(channel)

    # Normalize the activations.
    channel = (channel - a_min) / (a_max - a_min)
    a[:,:,i] = channel

    # Return normalized activations.
    return a


if __name__ == '__main__':
    with open('model.json') as f:
        j = json.load(f)

    w, b = j['w'], j['b']

    plot_model(w, b)
    plot_activations(w, b)
