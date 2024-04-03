import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gymnasium as gym
import numpy as np

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
    total_timesteps: int = 300000
    """total timesteps of the experiments"""
    learning_rate: float = 0.0001
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 64
    """the number of mini-batches"""
    update_epochs: int = 15
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    beta: float = 1
    """probability of expert actions inclusion in rollout buffer"""
    beta_decay: float = 0.008
    """decay rate of beta parameter"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

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

envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],)
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
agent = Agent(envs).to(device)

# Load the state dict from the .pt file
model_path = 'EI-PPO.pt'
state_dict = torch.load(model_path)

# Apply the loaded state dict to your model
agent.load_state_dict(state_dict)

harvest_list = []

for i in range(1):

    obs, _ = envs.reset(seed=args.seed)
    obs = torch.Tensor(next_obs).to(device)
    done = False

    while not done:
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(obs)

        #print(obs)
        #print("Action: ", action)
        next_obs, reward, done, truncations, infos = envs.step(action.cpu().numpy())
        #new_obs, reward, done, _, _ = env.step(action) 
        harvest = next_obs[0][11].item() * next_obs[0][10].item()
        
        obs = torch.Tensor(next_obs).to(device)  # Convert the next observation to tensor and move to the device
    
    #print(reward)
    #print("End")
    #print(obs)
    if(next_obs[0][7].item()==11):
        harvest_list.append(harvest)
        #print("Final yield: ", harvest)
    else:
        harvest_list.append(0)
        #print("Plant died")

    # return harvest_list

print(harvest_list)