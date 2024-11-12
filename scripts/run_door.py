from time_travel.envs.door import Action, DoorEnv
from time_travel.agents.door_agent import DoorAgent
from tqdm import tqdm

env = DoorEnv()
agent = DoorAgent(env)

for _ in tqdm(range(1000)):
    obs = env.reset()
    env_running = True

    rollout = []

    while env_running:
        primary_agent_action = None

        if env.is_original_timeline:
            normal_action = agent.act(obs[0])
            time_travel_action = None
            primary_agent_action = normal_action
        else:
            normal_action = agent.act(obs[0], deterministic=True)
            time_travel_action = agent.act(obs[1])
            primary_agent_action = time_travel_action

        prev_obs = obs
        obs, reward, terminated, truncated, info = env.step((normal_action, time_travel_action))
        env_running = not (terminated or truncated)

        rollout.append((prev_obs, primary_agent_action, reward, obs))

