from time_travel.envs.maze import Action, MazeEnv


env = MazeEnv()
obs = env.reset()
env_running = True
total_reward = 0

while env_running:
    env.render()

    print()
    normal_action = Action[input("Normal agent action: ")]
    time_travel_action = None if env.is_original_timeline else Action[input("Time travel action: ")]

    obs, reward, terminated, truncated, info = env.step((normal_action, time_travel_action))
    total_reward += reward

    env_running = not (terminated or truncated)
    print(f"\nReward {reward}\n")
    print(f"\nTotal reward: {total_reward}\n")
