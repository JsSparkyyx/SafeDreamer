import safety_gymnasium
import gymnasium as gym
import numpy as np
import torch

class SafeGymnasium:
    LOCK = None
    metadata = {}

    def __init__(
        self,
        name,
        action_repeat=1,
        size=(128, 128),
        # size=(256, 256),
        seed=None,
    ):
        assert size[0] == size[1]
        self._action_repeat = action_repeat
        self._size = size
        self._env = safety_gymnasium.make(name,render_mode='rgb_array',max_episode_steps=1100)
        print(self._env.observation_space)
        self._env.set_seed(seed)
        self._done = True
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        img_shape = self._size
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, img_shape + (3,), np.uint8),
                "vector": gym.spaces.Box(-np.inf, np.inf, shape=(40,), dtype=np.float32)
            }
        )

    def transform_obs(self, observation):
        obs = {}
        vectors = []
        for key in observation.keys():
            if key == "vision":
                obs[key] = observation[key]
            elif "vases" not in key and "hazards" not in key:
                vectors.append(observation[key].flatten())
        obs["vector"] = np.concatenate(vectors, axis=0)
        return obs

    @property
    def action_space(self):
        space = self._env.action_space
        return space

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            obs_dict, cur_reward, cost, terminated, truncated, info = self._env.step(action)
            # for key, value in obs_dict.items():
                # try:
                #     print(f"{key}: shape = {value.shape}")
                # except AttributeError:
                #     print(f"{key}: type = {type(value)}, value = {value}")
# vision: shape = (128, 128, 3)
# Reward: 0
# accelerometer: shape = (3,)
# velocimeter: shape = (3,)
# gyro: shape = (3,)
# magnetometer: shape = (3,)
# ballangvel_rear: shape = (3,)
# ballquat_rear: shape = (3, 3)
# goal_lidar: shape = (16,)
# hazards_lidar: shape = (16,)
# vases_lidar: shape = (16,)
            reward += cur_reward
            if terminated or truncated:
                break
        obs = {}
        risk = np.concatenate([obs_dict["vases_lidar"],obs_dict["hazards_lidar"]]).max()
        # if risk > 0.8:
        cost = -risk + 0.9
        # else:
            # cost = 0
        obs_dict = self.transform_obs(obs_dict)
        obs["image"] = obs_dict["vision"]
        obs["vector"] = obs_dict["vector"]
        obs["is_terminal"] = terminated or truncated
        obs["is_first"] = False
        info["cost"] = cost
        done = terminated or truncated
        return obs, cost, done, info

    def reset(self):
        obs_dict, info = self._env.reset()
        obs_dict = self.transform_obs(obs_dict)
        obs = {"is_terminal": False, "is_first": True}
        obs["image"] = obs_dict["vision"]
        obs["vector"] = obs_dict["vector"]
        info["cost"] = 0
        return obs

    def render(self, *args, **kwargs):
        return self._env.task.render(self._size[0], self._size[1], mode='rgb_array', camera_name='vision', cost={})

    def close(self):
        return self._env.close()
        

class SafeGymnasiumEval:
    LOCK = None
    metadata = {}

    def __init__(
        self,
        name,
        action_repeat=1,
        size=(128, 128),
        # size=(256, 256),
        seed=None,
    ):
        assert size[0] == size[1]
        self._action_repeat = action_repeat
        self._size = size
        self._env = safety_gymnasium.make(name,render_mode='rgb_array',max_episode_steps=1100)
        print(self._env.observation_space)
        self._env.set_seed(seed)
        self._done = True
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        img_shape = self._size
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, img_shape + (3,), np.uint8),
                "vector": gym.spaces.Box(-np.inf, np.inf, shape=(40,), dtype=np.float32)
            }
        )

    def transform_obs(self, observation):
        obs = {}
        vectors = []
        for key in observation.keys():
            if key == "vision":
                obs[key] = observation[key]
            elif "vases" not in key and "hazards" not in key:
                vectors.append(observation[key].flatten())
        obs["vector"] = np.concatenate(vectors, axis=0)
        return obs

    @property
    def action_space(self):
        space = self._env.action_space
        return space

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        cost = 0
        for _ in range(self._action_repeat):
            obs_dict, cur_reward, cur_cost, terminated, truncated, info = self._env.step(action)
            reward += cur_reward
            cost += cur_cost
            if terminated or truncated:
                break
        obs = {}
        obs_dict = self.transform_obs(obs_dict)
        obs["image"] = obs_dict["vision"]
        obs["vector"] = obs_dict["vector"]
        obs["is_terminal"] = terminated or truncated
        obs["is_first"] = False
        info["cost"] = cost
        done = terminated or truncated
        return obs, reward, done, info

    def reset(self):
        obs_dict, info = self._env.reset()
        obs_dict = self.transform_obs(obs_dict)
        obs = {"is_terminal": False, "is_first": True}
        obs["image"] = obs_dict["vision"]
        obs["vector"] = obs_dict["vector"]
        info["cost"] = 0
        return obs

    def render(self, *args, **kwargs):
        return self._env.task.render(self._size[0], self._size[1], mode='rgb_array', camera_name='vision', cost={})

    def close(self):
        return self._env.close()