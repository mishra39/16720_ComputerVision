from alignChannels import alignChannels
import numpy as np
import imageio

# Problem 1: Image Alignment

# 1. Load images (all 3 channels)
red = np.load('../data/red.npy', mmap_mode='r')
green = np.load('../data/green.npy', mmap_mode='r')
blue = np.load('../data/blue.npy', mmap_mode='r')

# 2. Find best alignment
rgbResult = alignChannels(red, green, blue)

# 3. save result to rgb_output.jpg (IN THE "results" FOLDER)
imageio.imsave('../results/rgb_output.jpg',rgbResult)