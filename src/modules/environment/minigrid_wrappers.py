from __future__ import annotations

import numpy as np
from gymnasium import spaces
from gymnasium.core import ObservationWrapper


class FullyObsWrapper(ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding instead of the agent view.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import FullyObsWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> obs['image'].shape
        (7, 7, 3)
        >>> env_obs = FullyObsWrapper(env)
        >>> obs, _ = env_obs.reset()
        >>> obs['image'].shape
        (11, 11, 3)
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.unwrapped.width, self.unwrapped.height, 3),
            dtype="uint8",
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = np.zeros((env.width, env.height, 4), dtype=np.uint8)
        full_grid[:, :, :3] = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1], 3] = env.agent_dir + 1

        return full_grid

    def getObservation(self):
        return self.observation(None)
