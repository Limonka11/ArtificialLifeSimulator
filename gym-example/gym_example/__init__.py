from gymnasium.envs.registration import register
import gymnasium as gym

# There will be a Python class Survival_Map defined within
# the envs subdirectory. When we register environment prior to
# training in RLib we'll use survival-map as its key.
register(
    id="survival-map-v0",
    entry_point="gym_example.envs:Survival_Map",
    max_episode_steps=300,
)