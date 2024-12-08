import gymnasium as gym
from gymnasium import spaces

from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
from time_travel.envs.maze import Action, MazeEnv, Observation

class MazeWrapper(gym.Wrapper):

    def __init__(self, env: MazeEnv, render_steps: bool = False):
        super().__init__(env)
        self.rollout = []
        self.agent = None
        self.render_steps = render_steps
        
    def set_agent(self, agent: RecurrentPPO):
        self.agent = agent

    def obs_to_array(self, obs: Observation):
        active_obs = obs[0] if self.env.is_original_timeline else obs[1]
        if active_obs is None:
            return [0, 0, 2, 2, 2, 2, 2, 0]
        return active_obs.to_array()

    def reset(self, seed=None, options=None):
        obs = self.env.reset(is_original_timeline=True)
        self.obs = self.obs_to_array(obs)
        return self.obs, None

    def step(self, action):
        # convert single action into joint action
        if self.render_steps:
            self.render()
            print(Action(value=action))

        if self.env.is_original_timeline:
            normal_action = action
            time_travel_action = None
        else:
            if self.obs == self.rollout[self.env.t][0]:
                normal_action = self.rollout[self.env.t][1]
            else:
                normal_action = self.agent.predict(self.obs, deterministic=True)
            time_travel_action = action

        prev_obs = self.obs
        joint_action = Action(value=normal_action), Action(value=time_travel_action) if time_travel_action is not None else None
        obs, reward, terminated, truncated, info = self.env.step(joint_action)
        self.obs = self.obs_to_array(obs)
        self.rollout.append((prev_obs, action, reward, self.obs))

        return self.obs, reward, terminated, truncated, info

maze_env = MazeEnv(trap_position_observed=False)
maze_wrapper = MazeWrapper(env=maze_env)

eval_env = MazeEnv(trap_position_observed=False)
eval_wrapper = MazeWrapper(env=eval_env, render_steps=True)
eval_callback = EvalCallback(eval_wrapper, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=1000,
                             deterministic=True, render=True)

rppo = RecurrentPPO("MlpLstmPolicy", maze_wrapper, verbose=1)
maze_wrapper.set_agent(rppo)
eval_wrapper.set_agent(rppo)
rppo.learn(int(1e5), callback=eval_callback)