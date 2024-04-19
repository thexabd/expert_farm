# Initialise the environment and add wrappers

from farmgym_games.game_builder.utils_sb3 import farmgym_to_gym_observations_flattened, wrapper
from farmgym_games.game_catalogue.farm0.farm import env as Farm0

env = Farm0()
orignal_obs, _  = env.reset()

# Wrap to change observation and action spaces and the step function
env.farmgym_to_gym_observations = farmgym_to_gym_observations_flattened
env = wrapper(env)
obs, _ = env.reset()

def expert_policy(obs):

    action = 0

    if obs[0] == 1:
        action = 6
    if obs[5] < 124:
        action = 1
    if obs[5] < 123:
        action = 2
    if obs[5] < 122:
        action = 3
    if obs[5] < 121:
        action = 4
    if obs[5] < 120:
        action = 5
    if obs[7] == 9:
        action = 7
    else:
        action = 6

    return action

def compute_returns(rewards, gamma=0.99):
    """
    Compute discounted cumulative rewards (returns) for an episode.
    """
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

# Expert dataset

from torch.utils.data.dataset import Dataset, random_split

class ExpertDataSetActor(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)
    
class ExpertDataSetCritic(Dataset):
    def __init__(self, expert_observations, expert_actions, expert_returns):
        self.observations = expert_observations
        self.actions = expert_actions
        self.returns = expert_returns

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index], self.returns[index])

    def __len__(self):
        return len(self.observations)
    
import numpy as np
import gym

# Function to generate offline data

def generate_offline_data(num_trajectories, gamma=0.99):
    expert_observations = []
    expert_actions = []
    expert_returns = []
    reward_threshold = 500

    trajectory_count = 0
    while trajectory_count < num_trajectories:
        obs, _ = env.reset()
        episode_rewards = []
        episode_observations = []
        episode_actions = []
        
        done = False
        while not done and trajectory_count < num_trajectories:
            action = expert_policy(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_observations.append(obs)
            episode_actions.append(action)
            episode_rewards.append(reward)

            obs = next_obs

        # Compute returns for the episode
        episode_returns = compute_returns(episode_rewards, gamma)
        
        if reward > reward_threshold:
            # Append episode data to expert data
            expert_observations.extend(episode_observations)
            expert_actions.extend(episode_actions)
            expert_returns.extend(episode_returns)
            trajectory_count += 1


    # Convert lists to numpy arrays
    expert_observations = np.array(expert_observations)
    expert_actions = np.array(expert_actions)
    expert_returns = np.array(expert_returns)

    # Save data to compressed file
    np.savez_compressed(
        "WS_data_{}".format(num_trajectories),
        expert_observations=expert_observations,
        expert_actions=expert_actions,
        expert_returns=expert_returns
    )

    return expert_observations, expert_actions, expert_returns

num_trajectories = 5
generate_offline_data(num_trajectories)

print("Successfully saved {} trajectories".format(num_trajectories))