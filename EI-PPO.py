import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

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

envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],)
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
agent = Agent(envs).to(device)

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