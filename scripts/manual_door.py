from time_travel.envs.door import Action, DoorEnv


env = DoorEnv()
obs = env.reset()
env_running = True

while env_running:
    env.render()

    print()
    normal_action = Action[input("Normal agent action: ")]
    time_travel_action = None if env.is_original_timeline else Action[input("Time travel action: ")]

    obs, reward, terminated, truncated, info = env.step((normal_action, time_travel_action))
    env_running = not (terminated or truncated)
    print(f"\nReward {reward}\n")
