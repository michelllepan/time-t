from time_travel.envs.door import Action, DoorEnv
from time_travel.agents.door_agent import DoorAgent
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def main():
    env = DoorEnv()
    agent = DoorAgent(env)

    total_rewards = []
    rollouts = []

    eval_rewards = []

    max_epsilon = 0.8
    max_episodes = 10000

    for episode_idx in tqdm(range(max_episodes)):
        obs = env.reset()
        env_running = True

        rollout = []
        total_reward = 0

        epsilon = max_epsilon * (1 - np.sqrt(episode_idx / max_episodes))

        while env_running:
            primary_agent_action = None
            prev_obs = None

            if env.is_original_timeline:
                normal_action = agent.act(obs[0], epsilon=epsilon, deterministic=False)
                time_travel_action = None
                primary_agent_action = normal_action
                prev_obs = obs[0]
            else:
                normal_action = agent.act(obs[0], deterministic=True)
                time_travel_action = agent.act(obs[1], epsilon=epsilon, deterministic=False)
                primary_agent_action = time_travel_action
                prev_obs = obs[1]

            obs, reward, terminated, truncated, info = env.step((normal_action, time_travel_action))
            env_running = not (terminated or truncated)

            curr_obs = obs[0] if env.is_original_timeline else obs[1]
            
            total_reward += reward
            rollout.append((prev_obs, primary_agent_action, reward, curr_obs))

            # if not env_running:
            #     print(f"success: {reward > 0}, time travel: {not env.is_original_timeline}")
            # print(f"t={info['t']} ({env_running=}): {rollout[-1]}")
        
        for s, a, r, sp in rollout:
            agent.update(s, a, sp, r)
        
        total_rewards.append(total_reward)
        rollouts.append(rollout)

        if episode_idx % 100 == 0:
            eval_rewards.append(eval(env, agent))
            print(f"Eval at {episode_idx=}: {eval_rewards[-1]}")

    for i in range(len(rollouts) - 5, len(rollouts)):
        print("\nNEW EPISODE")
        for step in rollouts[i]:
            print(step)
    # print(total_rewards)

    plt.xlabel("Episode")
    plt.ylabel("Evaluation reward")
    plt.plot(range(0, max_episodes, 100), eval_rewards)
    plt.savefig('eval_rew.png')
    


def eval(env, agent):
    total_reward = 0
    num_eval_episodes = 100
    for _ in tqdm(range(num_eval_episodes)):
        obs = env.reset()
        env_running = True

        episode_reward = 0

        while env_running:

            if env.is_original_timeline:
                normal_action = agent.act(obs[0], deterministic=True)
                time_travel_action = None
                primary_agent_action = normal_action
            else:
                normal_action = agent.act(obs[0], deterministic=True)
                time_travel_action = agent.act(obs[1], deterministic=True)
                primary_agent_action = time_travel_action

            if env.t == 0 and not env.is_original_timeline:
                print(f"lock action: {time_travel_action}")
            if env.t == 1:
                print(f"reward door: {env.reward_door}")
                print(f"normal agent action: {normal_action}")

            obs, reward, terminated, truncated, info = env.step((normal_action, time_travel_action))
            env_running = not (terminated or truncated)
            total_reward += reward
            episode_reward += reward

            
            if not env_running:
                print(f"success: {episode_reward > 0}, time travel: {not env.is_original_timeline}")
                print()

    return total_reward / num_eval_episodes


if __name__ == "__main__":
    main()