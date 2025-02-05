# Deep Reinforcement Learning for Satellite Image Segmentation and Classification

## Overview

This project applies deep reinforcement learning (DRL) to the segmentation and classification of satellite images. A Deep Q-Network (DQN) is trained in a custom environment to label patches of a satellite image (e.g., building, road, vegetation, water). While typical segmentation methods rely on supervised networks such as U-Net or Mask R-CNN, this project explores an alternative formulation that treats segmentation as a sequential decision-making problem.

## Methodology

- **Custom Environment**:  
  The environment simulates the segmentation task by dividing the input satellite image into non-overlapping patches. Each patch is paired with a ground-truth label (determined via majority vote from the corresponding segmentation mask). The agent selects a label for each patch and receives a reward (+1 for correct, -1 for incorrect).

- **Deep Q-Network (DQN)**:  
  A convolutional neural network (CNN) processes each image patch and outputs Q-values for each possible action (class label). An ε-greedy policy is used to balance exploration and exploitation during training.

- **Experience Replay**:  
  Transitions `(state, action, reward, next state, done)` are stored in a replay buffer. Mini-batches are randomly sampled during training to update the network parameters, which helps stabilize the learning process.

- **Inference and Visualization**:  
  After training, the agent labels each patch of the image. The predicted labels are reassembled into a full segmentation mask, which is then overlaid on the original image for visualization.

