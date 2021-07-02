import gym
import highway_env
import time
import numpy as np
from safe_explorer.core.config import Config

env = gym.make("highway-v0")
env.configure({
    "action": {
        "type": "ContinuousAction"
    }
})

class Highway(highway_env.highway_env.envs.HighwayEnv):
    def __init__(self):
        super(highway_env.highway_env.envs.HighwayEnv, self).__init__()
        self._config = Config.get().env.highway

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "absolute": False,
            },
            "action": {
                "type": "ContinuousAction",
            },
            "lanes_count": 2,
            "vehicles_count": 50,
        })
        return config

    def get_long_distance(self):
        '''returns longitudinal distance to closest car in front of ego vehicle'''
        # We define all the constraints such that C_i = 0
        # distance_closest > d_min => d_min - distance_closest < 0
        obs = self.observation_type.observe()
        x_pos, y_pos = obs[1:,1], obs[1:,2]     # array with car x and y positions, excluding ego vehicle
        x_pos = x_pos[np.absolute(y_pos) < 0.3] # only cars in the same lane as ego vehicle
        x_pos = x_pos[x_pos > 0] # only cars in front of ego vehicle
        if x_pos.size == 0: # if there is no car in front of the ego vehicle
            c = -1.0
        else:
            c = self._config.d_min - x_pos.min() # distance to closest car
        return c

    def get_num_constraints(self):
        return 1

    def get_constraint_values(self):
        long_dist_const = self.get_long_distance()
        return np.array(long_dist_const)
