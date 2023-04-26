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
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
import shutil
from ray.rllib.models import ModelCatalog
import numpy as np

from gym_example.envs.Entities.A2CBrain import DDQNBrain
from gym_example.envs.Entities.lstmBrain import TorchRNNModel

class SelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        # 0=RandomPolicy, 1=1st main policy snapshot,
        # 2=2nd main policy snapshot, etc..
        self.current_opponent = 0

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        """
        Used in order to add custom metrics to our tensorboard data
        """
        # Get env refernce from rllib wraper
        # env = base_env.get_unwrapped()[0]
        main_reward = 0
        other_reward = 0
        for key, value in episode.agent_rewards.items():
            print(key)
            if key[-1].startswith("gene_2") or key[-1].startswith("main_"):
                other_reward += value
            elif key[-1].startswith("main"):
                main_reward += value
            else:
                raise Exception("Sorry, expected only two teams")

        episode.custom_metrics["final_reward"] = main_reward
        episode.custom_metrics["win"] = 1 if main_reward > other_reward else 0
        episode.custom_metrics["draw"] = 1 if main_reward == other_reward else 0

    def on_train_result(self, *, algorithm, result, **kwargs):
        # Get the win rate for the train batch.
        # Note that normally, one should set up a proper evaluation config,
        # such that evaluation always happens on the already updated policy,
        # instead of on the already used train_batch.
        if "policy_main_reward" in result["hist_stats"]:
            #print("HERE: ", result["hist_stats"])
            # main_rew = result["hist_stats"].pop("policy_main_reward")
            # #opp_rew = result["hist_stats"].pop("policy_gene_2_policy_reward")
            # opponent_rew = list(result["hist_stats"].values())[0]
            # print("COMBINED: ",sum(opponent_rew))
            # assert len(main_rew) == len(opponent_rew)
            # won = 0
            # for r_main, r_opponent in zip(main_rew, opponent_rew):
            #     if r_main > r_opponent:
            #         won += 1

            win_rate = result["custom_metrics"]["win_mean"]
            result["win_rate"] = win_rate

            print(f"Iter={algorithm.iteration} win-rate={win_rate} -> ", end="")

            # If win rate is good -> Snapshot current policy and play against
            # it next, keeping the snapshot fixed and only improving the "main"
            # policy.
            if win_rate > 0.55: # win_rate_treshold
                self.current_opponent += 1
                new_pol_id = f"main_v{self.current_opponent}"
                print(f"Adding new opponent to the mix ({new_pol_id}).")

                # Re-define the mapping function, such that "main" is forced
                # to play against any of the previously played policies
                # (excluding "random").
                def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                    # agent_id = [0|1] -> policy depends on episode ID
                    # This way, we make sure that both policies sometimes play
                    # (start player) and sometimes agent1 (player to move 2nd).
                    #print("EPI: ", episode.episode_id % 2 == agent_id, agent_id)
                    return (
                        "main"
                        if agent_id.startswith("gene-1")
                        else "main_v{}".format(
                            np.random.choice(list(range(1, self.current_opponent + 1)))
                        )
                    )

                main_policy = algorithm.get_policy("main")
                # if algorithm.config._enable_learner_api:
                #     new_policy = algorithm.add_policy(
                #         policy_id=new_pol_id,
                #         policy_cls=type(main_policy),
                #         policy_mapping_fn=policy_mapping_fn,
                #         module_spec=SingleAgentRLModuleSpec.from_module(main_policy.model),
                #     )
                # else:
                new_policy = algorithm.add_policy(
                    policy_id=new_pol_id,
                    policy_cls=type(main_policy),
                    policy_mapping_fn=policy_mapping_fn,
                )

                # Set the weights of the new policy to the main policy.
                # We'll keep training the main policy, whereas `new_pol_id` will
                # remain fixed.
                main_state = main_policy.get_state()
                new_policy.set_state(main_state)

                # We need to sync the just copied local weights (from main policy)
                # to all the remote workers as well.
                algorithm.workers.sync_weights()
            else:
                print("not good enough; will keep learning ...")

            # +2 = main + random
            result["league_size"] = self.current_opponent + 2
        else:
            print("HERE:", result["hist_stats"])

def policy_mapper(agent_id, episode, worker):
    if agent_id.startswith("gene-1"):
        return "main"
    else:
        return "gene_2_policy"

def main():
    # init directory in which to save checkpoints
    chkpt_root = "tmp/exa"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


    ray.shutdown()
    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True)

    #print("Dashboard URL: http://{}".format(ray.get_webui_url()))

    # Register the custom model
    ModelCatalog.register_custom_model(
        "rnn", TorchRNNModel)
    
    # register the custom environment
    select_env = "survival-map-v0"
    env_config = {
        "size": 30,
        "brains": [DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain()],
        "max_agents": 20,
        "render_mode": "rgb_array"
    }
    register_env(select_env, lambda config: Survival_Map(config))

    # Configure the environment and create agent
    config_dict = {
        "env": select_env,
        "env_config": {
            "size": 30,
            "brains": [DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain()],
            "max_agents": 20,
            "render_mode": "rgb_array"
        },
        "gamma": 0.95,
        # These two may break stuff
        # https://github.com/ray-project/ray/issues/12709#issuecomment-741737472
        # https://chuacheowhuan.github.io/RLlib_trainer_config/
        "horizon": 2000,
        "soft_horizon": True,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 1,
        #change back to 20
        "num_envs_per_worker": 1,
        # Divide episodes into fragments of this many steps each during rollouts.
        # Sample batches of this size are collected from rollout workers and
        # combined into a larger batch of `train_batch_size` for learning.
        "rollout_fragment_length": 204,
        # Whether to rollout "complete_episodes" or "truncate_episodes" to
        # `rollout_fragment_length` length unrolls. Episode truncation guarantees
        # evenly sized batches, but increases variance as the reward-to-go will
        # need to be estimated at truncation boundaries.
        "batch_mode": "truncate_episodes",
        # Training batch size, if applicable. Should be >= rollout_fragment_length.
        # Samples batches will be concatenated together to a batch of this size,
        # which is then passed to SGD.
        "train_batch_size": 2048,
        #"entropy_coeff": 0.001,
        #"vf_loss_coeff": 1e-5,
        "lr": 0.0001,
        "model": {
            "custom_model": "rnn",
            "max_seq_len": 20,
            "custom_model_config": {
                "cell_size": 32,
            },
        },
        "framework": "torch",
        "multiagent": {
            # When you define the policy and map the agent_id to that policy, you
            # are defining the action and observation space for that agent.
            "policy_mapping_fn": policy_mapper,
            "policies": {"main", "gene_2_policy"},
            # Always just train the "main" policy.
            "policies_to_train": ["main"],
        },

        "callbacks": SelfPlayCallback,
    }

    algo = ppo.PPO(env=select_env, config=config_dict) 

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = 200

    train = False
    if train == True:
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


        # Examine one of the trained policies
        policy = algo.get_policy("main")
        model = policy.model
        print(model)

    if train == False:
        algo = ppo.PPO(env=select_env, config=config_dict)

    # Apply the trained policy in a rollout
    algo.restore("C:\\imperial\\MengProject\\saved_models\\checkpoint_000200")
    #algo.restore(chkpt_file)

    lstm_cell_size = config_dict["model"]["custom_model_config"]["cell_size"]

    env_config["render_mode"] = "human"
    env = Survival_Map(env_config)

    obs, infos = env.reset()
    sum_reward = 0
    n_step = 2000

    state_dict = {}
    init_state = {}
    for agent_id, agent_obs in obs.items():
        init_state[agent_id] = state_dict[agent_id] = [
            np.zeros([lstm_cell_size], np.float32) for _ in range(2)
        ]

    for step in range(n_step - 1):
        action = {}
        for agent_id, agent_obs in obs.items():
            state_dict[agent_id] = state_dict[agent_id] if agent_id in state_dict else [
            np.zeros([lstm_cell_size], np.float32) for _ in range(2)
        ]
            action[agent_id], state_dict[agent_id], _ = algo.compute_single_action(agent_obs, state=state_dict[agent_id], policy_id=policy_mapper(agent_id, 1, 1))
        obs, reward, done, truncated, info = env.step(action)
        sum_reward += sum(reward.values())

        env.render()

        if done["__all__"] == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            obs, infos = env.reset()
            state_dict = init_state
            sum_reward = 0



if __name__ == "__main__":
    #print(gym.envs.registry.keys())
    main()
    print("DONE")