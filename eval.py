import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gymnasium as gym
import numpy as np
import os
import tyro
import time
import argparse

from farmgym_games.game_builder.utils_sb3 import farmgym_to_gym_observations_flattened, wrapper
from farmgym_games.game_catalogue.farm0.farm import env as Farm0

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def make_env(env_id, idx, capture_video, run_name):

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = Farm0()
            orignal_obs, _  = env.reset()
            # Wrap to change observation and action spaces and the step function
            env.farmgym_to_gym_observations = farmgym_to_gym_observations_flattened
            env = wrapper(env)
            obs, _ = env.reset()
        # Recording statistics for analysis
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_loc', help='Location of the saved model')
    # Parse the command line arguments
    arg = parser.parse_args()

    #args = tyro.cli(Args)
    #run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    envs = gym.vector.SyncVectorEnv([make_env("Farm0", i, False, "run_name") for i in range(1)],)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the state dict from the .pt file
    agent = torch.load(arg.model_loc, map_location='cpu').to(device)

    harvest_list = []
    reward_list = []

    for i in range(100):

        obs, _ = envs.reset()
        obs = torch.Tensor(obs).to(device)
        done = False
        cum_reward = 0
        while not done:
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs)

            next_obs, reward, done, truncations, infos = envs.step(action.cpu().numpy())
    
            harvest = obs[0][11].item() * obs[0][10].item()
            cum_reward += reward.item()
            obs = torch.Tensor(next_obs).to(device)  # Convert the next observation to tensor and move to the device
        
        reward_list.append(cum_reward)

        if(action.cpu().numpy().item()==7):
            harvest_list.append(harvest)
            #print("Final yield: ", harvest)
        else:
            harvest_list.append(0)
            #print("Plant died")

    print("Reward list: ", reward_list)

    print(np.mean(reward_list), " +/- ", np.std(reward_list))

    print("Harvest list: ", harvest_list)

    print(np.mean(harvest_list), " +/- ", np.std(harvest_list))