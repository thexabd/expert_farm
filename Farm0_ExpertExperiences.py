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

from farmgym_games.game_builder.utils_sb3 import farmgym_to_gym_observations_flattened, wrapper
from farmgym_games.game_catalogue.farm0.farm import env as Farm0

from stable_baselines3 import PPO, A2C

import numpy as np
#from read import readExpertData

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
    total_timesteps: int = 500000
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
    beta_decay: float = 0.005
    """decay rate of beta parameter"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

env = Farm0()
orignal_obs, _  = env.reset()
# Wrap to change observation and action spaces and the step function
env.farmgym_to_gym_observations = farmgym_to_gym_observations_flattened
env = wrapper(env)
obs, _ = env.reset()

#expert_agent = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0001, n_epochs=15)
expert_agent = PPO.load("Expert_Agent_1m")

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

def expert_actions_values(model, next_obs):
    policy = model.policy
    #policy.device = model.policy.device
    policy.eval()

    next_obs = torch.tensor(next_obs).to(policy.device)

    with torch.no_grad():
        action_dist = policy.get_distribution(next_obs)
        logits = action_dist.distribution.logits
        action = torch.argmax(logits, dim=-1)
        # Evaluate the value and log probability of the chosen action
        value, log_prob, _ = policy.evaluate_actions(next_obs, action)
    
    return action, log_prob, 0, value

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
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Roll out buffer
    # ALGO Logic: Storage setup
    
    # Initialize the observation tensor to store observations for each step and environment.
    # The tensor is zero-initialized and shaped to accommodate the batch of observations from all environments over all steps.
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)

    # Initialize the actions tensor to store actions taken by the policy for each step and environment.
    # The tensor is zero-initialized and shaped to hold the batch of actions from all environments over all steps.
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)

    # Initialize the log probabilities tensor to store the log probabilities of the actions taken by the policy.
    # This is used later to compute the policy loss during optimization. It is zero-initialized.
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Initialize the rewards tensor to store the rewards received at each step for each environment.
    # This will be used to compute returns and advantages for the policy update. It is zero-initialized.
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Initialize the dones tensor to keep track of whether an episode has ended (done signal) at each step for each environment.
    # The done signal is important for resetting the environments and correctly calculating advantages. It is zero-initialized.
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Initialize the values tensor to store the value function estimates of the states as predicted by the value network.
    # These estimates are used to calculate the advantages for the policy update. It is zero-initialized.
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # EXPERT DATA
    # Load expert data from file data.txt
    # reader = readExpertData()
    # expert_obs, expert_actions, expert_rewards, expert_dones = reader.read_data_from_file('data.txt')

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Loop through the number of iterations defined for the training process
    for iteration in range(1, args.num_iterations + 1):
        # If learning rate annealing is enabled, adjust the learning rate based on the progress through the total iterations
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        beta_prob = random.random()

        # Iterate over each step in the rollout
        for step in range(0, args.num_steps):

            global_step += args.num_envs # Increment the global step count by the number of parallel environments
            obs[step] = next_obs # Record the current observation
            print(next_obs)
            #print("Obs: ", obs[step])
            dones[step] = next_done # Record whether the current state is a terminal state
            #print("Done: ", dones[step])

            # ALGO LOGIC: action logic
            # Disable gradient calculations for action selection as it's not part of the optimization process
            with torch.no_grad():
                # Obtain the action, log probability of the action, and value estimate from the policy network
                if beta_prob < args.beta: 
                    # Probability of including expert trajectories in the rollout buffer
                    action, logprob, _, value = expert_actions_values(expert_agent, next_obs)
                else:
                    # Obtain the action, log probability of the action, and value estimate from the policy network
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    
                values[step] = value.flatten()  # Flatten the value tensor for storage
                actions[step] = action  # Store the action
                logprobs[step] = logprob  # Store the log probability of the action
                #print("Actions: ", actions[step])

            # TRY NOT TO MODIFY: execute the game and log data.
            # Execute the action in the environment and log the results
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)  # Determine if the episode is done either by termination or truncation
            rewards[step] = torch.tensor(reward).to(device).view(-1)  # Store the reward and move it to the device (e.g., GPU)
            #print("Rewards: ", rewards[step])
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)  # Convert the next observation and done signal to tensors and move to the device

            # If there are final info objects, which contain episodic summary data, log them
            if "final_info" in infos:
                beta_prob = random.random()
                #print(beta_prob)

                for info in infos["final_info"]:
                    if info and "episode" in info:
                        # Print and log episodic return and length information using TensorBoard
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        
        #print(args.beta)
        args.beta -= args.beta_decay
        
        # Bootstrap value if not done
        # Use no gradient tracking for efficiency since this is only for inference, not training
        with torch.no_grad():
            # Get the estimated value of the next state from the value network (critic)
            next_value = agent.get_value(next_obs).reshape(1, -1)
            # Prepare the tensor that will hold the advantages, initialized to zeros
            advantages = torch.zeros_like(rewards).to(device)
            # Initialize the variable that will hold the last GAE (Generalized Advantage Estimation) lambda
            lastgaelam = 0

            # Reverse iteration through each step to compute GAE and returns
            for t in reversed(range(args.num_steps)):
                # If processing the last timestep, nextnonterminal is 1 if the state is not done
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                     # For all other timesteps, nextnonterminal is 1 if the next state is not done
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                # Calculate the temporal difference error (delta) for the current step
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                # Update the advantages using GAE formula: delta + discount * lambda * nextnonterminal * previous advantage
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            # The returns are the sum of the advantages and the value estimates
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        # Initialize batch indices array for randomized mini-batch sampling
        b_inds = np.arange(args.batch_size)
        # List to store the fraction of policy probabilities that were clipped
        clipfracs = []

        # Optimization loop over the policy and value network for a number of epochs
        for epoch in range(args.update_epochs):
            # Shuffle the batch indices to sample random mini-batches
            np.random.shuffle(b_inds)
            # Iterate over the batch in mini-batch increments
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                # Slice the batch indices to obtain a mini-batch
                mb_inds = b_inds[start:end]

                # Get the predicted log probabilities, entropy, and values for the current mini-batch
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                # Calculate the log differences between new and old probabilities
                logratio = newlogprob - b_logprobs[mb_inds]
                # Compute the ratio of new to old probabilities
                ratio = logratio.exp()

                # Calculate the KL divergence and the clipping fraction for diagnostics
                with torch.no_grad():
                    # Estimate the KL divergence before the policy update
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    # Estimate the KL divergence after the policy update
                    approx_kl = ((ratio - 1) - logratio).mean()
                    # Calculate the fraction of probabilities that were clipped
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # Select the advantages corresponding to the current mini-batch
                mb_advantages = b_advantages[mb_inds]
                # If enabled, normalize advantages
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                # Calculate the two forms of the policy loss: the unclipped and the clipped
                pg_loss1 = -mb_advantages * ratio # Unclipped policy gradient loss
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) # Clipped policy gradient loss
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() # Final policy loss is the mean of the maximum of pg_loss1 and pg_loss2

                # Value loss
                newvalue = newvalue.view(-1) # Ensure the new value predictions have the correct shape
                if args.clip_vloss:
                    # If clipping the value loss is enabled
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2 # Squared differences between new value predictions and the actual returns
                    v_clipped = b_values[mb_inds] + torch.clamp( # Clipped value predictions
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2 # Squared differences for the clipped value predictions
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped) # Take the maximum of unclipped and clipped value losses
                    v_loss = 0.5 * v_loss_max.mean() # The value loss is the mean of v_loss_max scaled by 0.5
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean() # If not clipping, use simple squared differences

                # Calculate the entropy loss to encourage exploration by penalizing certainty
                entropy_loss = entropy.mean()

                # Total loss is the sum of policy loss, value loss (scaled by vf_coef), and entropy loss (scaled by ent_coef)
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # Prepare for the gradient update
                optimizer.zero_grad() # Zero out any existing gradients
                loss.backward() # Backpropagate the loss to calculate gradients
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm) # Clip gradients to avoid large updates
                optimizer.step() # Perform a single optimization step

            # Check if the KL divergence between the new and old policy exceeds a threshold for early stopping
            if args.target_kl is not None and approx_kl > args.target_kl:
                break # Stop the optimization loop if KL divergence is too large

        # Convert PyTorch tensors to NumPy arrays for predicted values and actual returns
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        
        # Calculate the variance of the actual returns to measure how much the returns vary
        var_y = np.var(y_true)

        # Compute the explained variance, which indicates how well the predicted values approximate the actual returns
        # If the variance of actual returns is zero (all returns are the same), explained variance is not applicable (set to NaN)
        # Otherwise, explained variance is calculated as 1 minus the ratio of the variance of the residuals (actual minus predicted)
        # to the variance of actual returns, which represents the proportion of return variance explained by the model
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/beta", args.beta, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

    # Write a method to calculate harvest for 100 episodes
    # Track decrease of beta parameter throughout the learning process