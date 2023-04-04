import sys
sys.path.insert(0, "C:\\imperial\\MengProject\\gym-example")
sys.path.insert(0, "C:\\imperial\\MengProject\\gym-example\\gym_example")
from gym_example.envs.Survival_Map import Survival_Map
#from gym_example.envs.fail1 import Fail_v1
from ray.tune.registry import register_env
import gymnasium as gym
import os
import ray
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
import shutil


def main():
    # init directory in which to save checkpoints
    chkpt_root = "tmp/exa"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True)

    # register the custom environment
    select_env = "survival-map-v0"
    #select_env = "fail-v1"
    register_env(select_env, lambda config: Survival_Map())
    #register_env(select_env, lambda config: Fail_v1())


    # configure the environment and create agent
    config = PPOConfig()  
    config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3)  
    config = config.resources(num_gpus=0)  
    config = config.rollouts(num_rollout_workers=4)
    algo = ppo.PPO(env=select_env, config=config) 

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = 5

    # train a policy with RLlib using PPO
    for n in range(n_iter):
        # Run the episodes
        result = algo.train()
        # Save a checkpoint of the latest policy
        chkpt_file = algo.save(chkpt_root)

        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                chkpt_file
                ))


    # examine the trained policy
    policy = algo.get_policy()
    model = policy.model
    print(model.base_model.summary())


    # apply the trained policy in a rollout
    algo.restore(chkpt_file)
    env = gym.make("survival-map-v0", size = 50, render_mode="human")

    state = env.reset()
    sum_reward = 0
    n_step = 20

    for step in range(n_step - 1):
        print("HERE", type(state))
        if type(state) is tuple:
            state = state[0]
        action = algo.compute_single_action(state)
        state, reward, done, truncated, info = env.step(action)
        sum_reward += reward

        env.render()

        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0


if __name__ == "__main__":
    #print(gym.envs.registry.keys())
    main()