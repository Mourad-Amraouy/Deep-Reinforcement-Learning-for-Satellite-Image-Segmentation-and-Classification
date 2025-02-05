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

## Modified Code with `patch_size = 2500`

> **Note:**  
> With `patch_size = 2500` and an image size of 2500×2500, the entire image is treated as one patch. For multiple patches, use a larger image (e.g., 5000×5000). Also, a large patch size results in a very high-dimensional feature vector after the convolutional layers, which can increase memory and computation requirements.

Below is the complete Python code:

```python
import numpy as np
import gym
from gym import spaces
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque

##########################################
# 1. Define the Custom Environment
##########################################

class SatelliteSegmentationEnv(gym.Env):
    """
    A simplified Gym-like environment for patch-based segmentation.
    
    The image is divided into non-overlapping patches (of size patch_size x patch_size).
    Each patch is assigned a ground-truth label (by majority vote in the corresponding
    region of the ground-truth mask). The agent observes one patch at a time and chooses
    an action (label). The reward is +1 if the label is correct, else -1.
    """
    def __init__(self, image, gt_mask, patch_size=2500, num_classes=4):
        super(SatelliteSegmentationEnv, self).__init__()
        self.image = image            # e.g. shape (H, W, 3)
        self.gt_mask = gt_mask        # e.g. shape (H, W) with values {0,1,2,3}
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.patches = []             # list of image patches
        self.gt_labels = []           # corresponding ground-truth labels for each patch
        self.build_patches()
        self.total_patches = len(self.patches)
        self.current_patch = 0
        # The agent’s action is to choose one of the num_classes labels.
        self.action_space = spaces.Discrete(num_classes)
        # Observation: the current patch (with pixel values 0-255)
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(patch_size, patch_size, 3), dtype=np.uint8)
    
    def build_patches(self):
        H, W, _ = self.image.shape
        self.num_patches_h = H // self.patch_size
        self.num_patches_w = W // self.patch_size
        for i in range(self.num_patches_h):
            for j in range(self.num_patches_w):
                # Extract image patch
                patch = self.image[i*self.patch_size:(i+1)*self.patch_size,
                                   j*self.patch_size:(j+1)*self.patch_size, :]
                # Extract the corresponding part of the ground-truth mask
                patch_gt = self.gt_mask[i*self.patch_size:(i+1)*self.patch_size,
                                         j*self.patch_size:(j+1)*self.patch_size]
                # Use majority vote to decide the patch's true label
                label = np.bincount(patch_gt.flatten()).argmax()
                self.patches.append(patch)
                self.gt_labels.append(label)
        # Convert lists to numpy arrays
        self.patches = np.array(self.patches)
        self.gt_labels = np.array(self.gt_labels)
    
    def reset(self):
        self.current_patch = 0
        # Initialize predicted labels with -1 (unknown)
        self.pred_labels = -1 * np.ones_like(self.gt_labels)
        return self._get_obs()
    
    def _get_obs(self):
        if self.current_patch < self.total_patches:
            return self.patches[self.current_patch]
        else:
            return None
    
    def step(self, action):
        """
        Take an action (i.e. assign a label to the current patch).
        Returns: next_state, reward, done, info
        """
        done = False
        true_label = self.gt_labels[self.current_patch]
        reward = 1.0 if action == true_label else -1.0
        self.pred_labels[self.current_patch] = action
        self.current_patch += 1
        if self.current_patch >= self.total_patches:
            done = True
        next_state = self._get_obs() if not done else None
        return next_state, reward, done, {}

##########################################
# 2. Define the DQN Network
##########################################

class DQN(nn.Module):
    """
    A simple convolutional network that takes an image patch and outputs Q-values
    for each possible action (i.e. class label).
    """
    def __init__(self, patch_size, num_classes):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # halves the spatial dimensions
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # halves dimensions again
        )
        # Compute the flattened feature size after convolution
        conv_out_size = (patch_size // 4) * (patch_size // 4) * 32
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Expect x shape: (batch, patch_size, patch_size, 3)
        x = x.permute(0, 3, 1, 2)  # convert to (batch, 3, patch_size, patch_size)
        x = x / 255.0              # normalize pixel values to [0,1]
        x = self.conv(x)
        # Using reshape instead of view to handle non-contiguous tensors
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

##########################################
# 3. Experience Replay Buffer
##########################################

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

##########################################
# 4. Training Setup and Loop
##########################################

# Hyperparameters
num_classes = 4           # building, road, vegetation, water
patch_size = 2500         # size of each patch is now 2500
num_episodes = 1000       # number of training episodes
batch_size = 32
gamma = 0.99              # discount factor
learning_rate = 1e-3
epsilon_start = 1.0       # starting value for epsilon in epsilon-greedy
epsilon_final = 0.1
epsilon_decay = 500       # decay factor (adjust as needed)

# For demonstration, we simulate a satellite image and its ground truth segmentation.
# To support patch_size=2500, we create an image of size 2500 x 2500.
H, W = 2500, 2500
image = np.random.randint(0, 256, size=(H, W, 3), dtype=np.uint8)
gt_mask = np.random.randint(0, num_classes, size=(H, W), dtype=np.uint8)

# Create the environment and the DQN agent
env = SatelliteSegmentationEnv(image, gt_mask, patch_size, num_classes)
dqn = DQN(patch_size, num_classes)
optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
memory = ReplayBuffer(10000)

def get_epsilon(steps_done):
    """Exponential decay of epsilon for exploration."""
    return epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * steps_done / epsilon_decay)

steps_done = 0

print("Starting training ...")
for episode in range(num_episodes):
    state = env.reset()  # reset the environment; state is the first patch
    episode_reward = 0
    done = False
    while not done:
        epsilon = get_epsilon(steps_done)
        steps_done += 1
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float32)  # shape: (1, patch_size, patch_size, 3)
            with torch.no_grad():
                q_values = dqn(state_tensor)
            action = q_values.argmax().item()
        
        next_state, reward, done, _ = env.step(action)
        memory.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        
        # Perform an optimization step if enough transitions are available
        if len(memory) >= batch_size:
            states_b, actions_b, rewards_b, next_states_b, dones_b = memory.sample(batch_size)
            states_tensor = torch.tensor(np.array(states_b), dtype=torch.float32)  # (batch, patch_size, patch_size, 3)
            actions_tensor = torch.tensor(actions_b, dtype=torch.int64).unsqueeze(1)
            rewards_tensor = torch.tensor(rewards_b, dtype=torch.float32)
            
            # Create a mask for non-final next states
            non_final_mask = torch.tensor([s is not None for s in next_states_b], dtype=torch.bool)
            if non_final_mask.sum() > 0:
                non_final_next_states = torch.tensor(
                    np.array([s for s in next_states_b if s is not None]),
                    dtype=torch.float32)
            else:
                non_final_next_states = torch.empty((0, patch_size, patch_size, 3))
            
            # Compute current Q values
            q_values = dqn(states_tensor).gather(1, actions_tensor).squeeze(1)
            next_q_values = torch.zeros(batch_size)
            if non_final_next_states.shape[0] > 0:
                next_q = dqn(non_final_next_states).max(1)[0].detach()
                next_q_values[non_final_mask] = next_q
            target_q_values = rewards_tensor + gamma * next_q_values * (1 - torch.tensor(dones_b, dtype=torch.float32))
            
            loss = F.mse_loss(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Print status every 100 episodes
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}")

##########################################
# 5. Inference & Visualization
##########################################

# After training, have the agent segment the image.
state = env.reset()
predicted_labels = []
done = False
while not done:
    state_tensor = torch.tensor(np.array([state]), dtype=torch.float32)
    with torch.no_grad():
        q_values = dqn(state_tensor)
    action = q_values.argmax().item()
    predicted_labels.append(action)
    state, reward, done, _ = env.step(action)

# Reshape the predicted labels into a 2D grid (one label per patch)
patches_h = env.num_patches_h
patches_w = env.num_patches_w
predicted_mask = np.array(predicted_labels).reshape((patches_h, patches_w))

# Upsample the patch-level segmentation to the original image size (using np.kron)
segmented_mask = np.kron(predicted_mask, np.ones((patch_size, patch_size)))

# Plot the original image and the segmentation overlay
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(env.image)
plt.title("Original Satellite Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(env.image)
plt.imshow(segmented_mask, cmap='jet', alpha=0.5)
plt.title("Segmented & Classified Image")
plt.axis('off')

plt.show()
