"""
This is a simple training script that uses the PPO algorithm to train a policy.
Most of the code are from Cory Binnersley's Google Colab notebook.
If you want to visit the original notebook, please visit the following link:
https://colab.research.google.com/drive/19Rz916XaYRlq9sOgi8VtXdHgOMkysw2M?usp=sharing
"""
from gym_battlesnake.custompolicy import SnakePolicyBase, create_policy
import numpy as np
import os
import time
import torch
from datetime import datetime

from a2c_ppo_acktr.algo import PPO
from a2c_ppo_acktr.storage import RolloutStorage
from gym_battlesnake.gymbattlesnake import BattlesnakeEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_date = datetime.now().strftime("%Y%m%d")

# Number of parallel environments to generate games in
n_envs = 150
# Number of steps per environment to simulate
n_steps = 500

# The total training set size per iteration is n_envs * n_steps

# The gym environment
env = BattlesnakeEnv(n_threads=4, n_envs=n_envs)

# Storage for rollouts (game turns played and the rewards)
rollouts = RolloutStorage(n_steps,
                          n_envs,
                          env.observation_space.shape,
                          env.action_space,
                          n_steps)
env.close()

# Create our policy as defined above
policy = create_policy(env.observation_space.shape, env.action_space, SnakePolicyBase)
best_old_policy = create_policy(env.observation_space.shape, env.action_space, SnakePolicyBase)
random_best_old_policy = create_policy(env.observation_space.shape, env.action_space, SnakePolicyBase)

# Lets make the old policy the same as the current one
policy.load_state_dict(policy.state_dict())
random_best_old_policy.load_state_dict(policy.state_dict())
best_old_policy.load_state_dict(policy.state_dict())

# TODO: if you already have a saved model, you can load it here
#LOAD A SAVED MODEL
#policy.load_state_dict(torch.load('example_model.pth'))
#best_old_policy.load_state_dict(torch.load('example_model.pth'))

agent = PPO(policy,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            clip_param=0.2,
            ppo_epoch=4,
            num_mini_batch=16,
            eps=1e-5,
            lr=5e-5)

# Let's define a method to check our performance against an older policy
# Determines an unbiased winrate check
# TODO: If you want to change the number of opponents, you can change the n_opponents parameter
def check_performance(current_policy, opponent, n_opponents=1, n_envs=1000, steps=1500, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    test_env = BattlesnakeEnv(n_threads=os.cpu_count(), n_envs=n_envs, opponents=[opponent for _ in range(n_opponents)], device=device)
    obs = test_env.reset()
    wins = 0
    losses = 0
    completed = np.zeros(n_envs)
    count = 0
    lengths = []
    with torch.no_grad():
        # Simulate to a maximum steps across our environments, only recording the first result in each env.
        print("Running performance check")
        for step in tqdm(range(steps)):
            if count == n_envs:
                # Quick break
                print("Check Performance done @ step", step)
                
                break
            inp = torch.tensor(obs, dtype=torch.float32).to(device)
            action, _ = current_policy.predict(inp, deterministic=True)
            obs, reward, done, info = test_env.step(action.cpu().numpy().flatten())
            for i in range(test_env.n_envs):
                if completed[i] == 1:
                    continue # Only count each environment once
                if 'episode' in info[i]:
                    if info[i]['episode']['r'] == 1:
                        completed[i] = 1
                        count += 1
                        wins += 1
                        lengths.append(info[i]['episode']['l'])
                    elif info[i]['episode']['r'] == -1:
                        completed[i] = 1
                        losses += 1
                        count += 1
                        lengths.append(info[i]['episode']['l'])

    winrate = wins / n_envs
    print("Wins", wins)
    print("Losses", losses)
    print("Average episode length:", np.mean(lengths))
    return winrate

from tqdm.notebook import tqdm
# We'll play 2-way matches
#TODO: If you want to change the number of opponents, you can change the parameter of opponents in range()
env = BattlesnakeEnv(n_threads=8, n_envs=n_envs, opponents=[policy for _ in range(1)], device=device)
obs = env.reset()
rollouts.obs[0].copy_(torch.tensor(obs))

# How many iterations do we want to run
num_updates = 2500

#save_path
save_path = 'models'
save_best_path = "best_model"

# Send our network and storage to the gpu
policy.to(device)
best_old_policy.to(device)
random_best_old_policy.to(device)
rollouts.to(device)

# Record mean values to plot at the end
rewards = []
value_losses = []
lengths = []

best_one_num = 0
model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
random_model_file = np.random.choice(model_files) if model_files else "No models found"

state_dict_count = 1
state_dict_name = f'state_dict_{current_date}_{state_dict_count:03d}.pth'
state_dict_path = os.path.join(save_path, state_dict_name)

model_count = 1
model_name = f'best_model{model_count:03d}.pth'
model_path = os.path.join(save_best_path, model_name)

start = time.time()
for j in range(num_updates):
    episode_rewards = []
    episode_lengths = []
    # Set
    policy.eval()
    print(f"Iteration {j+1}: Generating rollouts")
    for step in tqdm(range(n_steps)):
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = policy.act(rollouts.obs[step],
                                                            rollouts.recurrent_hidden_states[step],
                                                            rollouts.masks[step])
        obs, reward, done, infos = env.step(action.cpu().squeeze())
        obs = torch.tensor(obs)
        reward = torch.tensor(reward).unsqueeze(1)

        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
                episode_lengths.append(info['episode']['l'])

        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
        rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

    with torch.no_grad():
        next_value = policy.get_value(
            rollouts.obs[-1],
            rollouts.recurrent_hidden_states[-1],
            rollouts.masks[-1]
        ).detach()
        
    # Set the policy to be in training mode (switches modules to training mode for things like batchnorm layers)
    policy.train()

    print("Training policy on rollouts...")
    # We're using a gamma = 0.99 and lambda = 0.95
    rollouts.compute_returns(next_value, True, 0.99, 0.95, False)
    value_loss, action_loss, dist_entropy = agent.update(rollouts)
    rollouts.after_update()

    # Set the policy into eval mode (for batchnorms, etc)
    policy.eval()
    
    total_num_steps = (j + 1) * n_envs * n_steps
    end = time.time()
    
    lengths.append(np.mean(episode_lengths))
    rewards.append(np.mean(episode_rewards))
    value_losses.append(value_loss)
    
    # Every 10 iterations, we'll print out the episode metrics
    if (j+1) % 50 == 0:
        if model_files:
            random_model_file = np.random.choice(model_files)
            random_model_path = os.path.join('models', random_model_file)
            random_best_old_policy.load_state_dict(torch.load(random_model_path))
        else:
            print("No models found in models/ directory. Using current policy as random best policy")
            
        print("\n")
        print("=" * 80)
        print("Iteration", j+1, "Results")
        # Check the performance of the current policy against the prior best
        winrate1 = check_performance(policy, best_old_policy, device=torch.device(device))
        winrate2 = check_performance(policy, random_best_old_policy, device=torch.device(device))
        print(f"Winrate vs prior best({best_one_num}): {winrate1*100:.2f}%")
        print(f"Winrate vs random best({random_model_file}): {winrate2*100:.2f}%")
        print(f"Median Length: {np.median(episode_lengths)}")
        print(f"Max Length: {np.max(episode_lengths)}")
        print(f"Min Length: {np.min(episode_lengths)}")
        
        #save as TorchScript 
        #model_scripted = torch.jit.script(policy)
        #model_scripted.save(os.path.join(save_path, f'model_scripted_{j+1}.pth'))

        # If our policy wins more than 53% of the games against the prior
        # best opponent, update the prior best.
        # TODO: If you play against different number of opponents, you can change the winrate1 and winrate2 parameters
        # TODO: Or you can change the winrate value as you want
        if winrate1 >= 0.53 and winrate2 >= 0.53:
            print("Policy winrate is > 53%. Updating prior best model...")
            best_old_policy.load_state_dict(policy.state_dict())
            best_one_num = j+1
            #save best model's state_dict
            while os.path.exists(state_dict_path):
                state_dict_count += 1
                state_dict_name = f'state_dict_{current_date}_{state_dict_count:03d}.pth'
                state_dict_path = os.path.join(save_path, state_dict_name)
            torch.save(best_old_policy.state_dict(), state_dict_path)
            #save best model
            while os.path.exists(model_path):
                model_count += 1
                model_name = f'best_model{model_count:03d}.pth'
                model_path = os.path.join(save_best_path, model_name)
            torch.save(best_old_policy, model_path)
            model_files.append(state_dict_name)
            while len(model_files) > 5:
                model_files.remove(model_files[0])
        else:
            print("Policy has not learned enough yet... keep training!")
        print("-" * 80)
        