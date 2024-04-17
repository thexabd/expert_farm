# EXPERT INITIATED PPO

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from farmgym_games.game_builder.utils_sb3 import farmgym_to_gym_observations_flattened, wrapper
from farmgym_games.game_catalogue.farm0.farm import env as Farm0

import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Farm0"
    """the id of the environment"""
    num_envs: int = 1
    """the number of parallel game environments"""
    learning_rate: float = 0.0001
    """learning rate"""

env = Farm0()
orignal_obs, _  = env.reset()
# Wrap to change observation and action spaces and the step function
env.farmgym_to_gym_observations = farmgym_to_gym_observations_flattened
env = wrapper(env)
obs, _ = env.reset()

# Making the environment
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

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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

class ExpertDatasetActor(Dataset):
    def __init__(self, observations, actions):
        self.observations = observations
        self.actions = actions
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]
    
class ExpertDatasetCritic(Dataset):
    def __init__(self, observations, returns):
        self.observations = observations
        self.returns = actions
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.returns[idx]
    
if __name__ == "__main__":
    args = tyro.cli(Args)
  
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Creating environment
    # Env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],)
    # Check if environment has a discrete action space
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Put agent to device and set optimizer
    agent = Agent(envs).to(device)
    optimizer1 = optim.Adam(agent.parameters(), lr=args.learning_rate)
    optimizer2 = optim.Adam(agent.parameters(), lr=0.1)

    # Load the NPZ file
    data = np.load("WS_data_1000.npz")
    observations = data['expert_observations']  
    actions = data['expert_actions']  
    returns = data['expert_returns']

    # Prepare the data by converting to torch tensors
    observations = torch.tensor(observations, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    returns = torch.tensor(returns, dtype=torch.float32)

    # Create dataset object
    expert_dataset_actor = ExpertDatasetActor(observations, actions)
    expert_dataset_critic = ExpertDatasetCritic(observations, returns)

    # Create DataLoader for batching
    expert_data_loader_actor = DataLoader(expert_dataset_actor, batch_size=64, shuffle=True)
    expert_data_loader_critic = DataLoader(expert_dataset_critic, batch_size=64, shuffle=True)

    # Modify training loop for behavior cloning
    def train_actor(agent, data_loader, optimizer, epochs):
        mean_loss = []
        for epoch in range(epochs):
            mean_loss_epoch = []
            for observations, expert_actions in data_loader:
                observations = observations.to(device)
                expert_actions = expert_actions.to(device)
                predicted_actions = agent.actor(observations)
                loss = F.cross_entropy(predicted_actions, expert_actions)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mean_loss_epoch.append(loss.item())
                print(f"Epoch {epoch}, Actor Loss: {loss.item()}")
            mean_loss.append(sum(mean_loss_epoch)/len(mean_loss_epoch))
        
        return mean_loss
    
    def train_critic(agent, data_loader, optimizer, epochs):
        mean_loss = []
        for epoch in range(epochs):
            mean_loss_epoch = []
            for observations, expert_returns in data_loader:
                observations = observations.to(device)
                expert_returns = expert_returns.to(device).float()
                predicted_values = agent.critic(observations)
                predicted_values = predicted_values.squeeze()
                loss = F.cross_entropy(predicted_values, expert_returns)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mean_loss_epoch.append(loss.item())
                print(f"Epoch {epoch}, Critic Loss: {loss.item()}")
            mean_loss.append(sum(mean_loss_epoch)/len(mean_loss_epoch))
        
        return mean_loss

    actor_loss = train_actor(agent, expert_data_loader_actor, optimizer1, epochs=20)
    critic_loss = train_critic(agent, expert_data_loader_critic, optimizer2, epochs=20)
    
    plt.plot([i+1 for i in range(0, 20, 1)], actor_loss, label="Actor Train Loss")
    plt.legend()
    plt.show()

    plt.plot([i+1 for i in range(0, 20, 1)], critic_loss, label="Critic Train Loss")
    plt.legend()
    plt.show()

    torch.save(agent, 'ws.pt')