import sys
sys.path.insert(0, "C:\\imperial\\MengProject\\gym-example")
sys.path.insert(0, "C:\\imperial\\MengProject\\gym-example\\gym_example")
from gym_example.envs.Survival_Map import Survival_Map
#from gym_example.envs.fail1 import Fail_v1
from ray.tune.registry import register_env
import gymnasium as gym
import os
import ray
from ray import air, tune
from ray.rllib.algorithms import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
import shutil
from ray.rllib.models import ModelCatalog
import numpy as np
import time

from gym_example.envs.Entities.A2CBrain import DDQNBrain
from gym_example.envs.Entities.lstmBrain import TorchRNNModel

# class SelfPlayCallback(DefaultCallbacks):
#     def __init__(self):
#         super().__init__()
#         # 0=RandomPolicy, 1=1st main policy snapshot,
#         # 2=2nd main policy snapshot, etc..
#         self.current_opponent = 0
#         self.last_removed_opponent = 1
#         self.update_fn = False

#         self.policy_names = set()
#         #self.policy_names.add("main")
#         self.policy_names.add("gene_2_policy")

#     def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
#         """
#         Used in order to add custom metrics to our tensorboard data
#         """
#         # Get env refernce from rllib wraper
#         # env = base_env.get_unwrapped()[0]
#         #policy_ids = [p for p in policies.keys()]
            
#         # if self.update_fn:
#         #     # inplace reset of the worker context which is shared across various parts and the environment
#         #     # which is accessed by workers sample collector and related parts
#         #     def reset_env(env, ctx):
#         #         env.reset()

#         #     # Per worker policy map fn updates - note in reduced form of problem no extra code was added
#         #     # to handle oddities with global worker for local mode
#         #     def policy_mapping_fn(agent_id, episode, worker, **kwargs):
#         #         #print("EPI: ", episode.episode_id % 2 == agent_id, agent_id)
#         #         gay = np.random.randint(worker.callbacks.last_removed_opponent, self.current_opponent + 1)
#         #         print("GAY2: ", gay, " ", list(range(worker.callbacks.last_removed_opponent, worker.callbacks.current_opponent + 1)))
#         #         return (
#         #             "main"
#         #             if agent_id.startswith("gene-1")
#         #             else "main_v{}".format(
#         #                 gay
#         #             )
#         #         )

#         #     worker.set_policy_mapping_fn(policy_mapping_fn)
#         #     #worker.set_policies_to_train(["main"])
#         #     worker.foreach_env_with_context(reset_env)

#         #     self.update_fn = False

#         main_reward = 0
#         other_reward = 0
#         for key, value in episode.agent_rewards.items():
#             #print(key)
#             if key[-1].startswith("gene_2") or key[-1].startswith("main_"):
#                 other_reward += value
#             elif key[-1].startswith("main"):
#                 main_reward += value
#             else:
#                 raise Exception("Sorry, expected only two teams")

#         episode.custom_metrics["final_reward"] = main_reward
#         episode.custom_metrics["win"] = 1 if main_reward > other_reward else 0
#         episode.custom_metrics["draw"] = 1 if main_reward == other_reward else 0

#     def on_train_result(self, *, algorithm, result, **kwargs):
#         # Get the win rate for the train batch.
#         # Note that normally, one should set up a proper evaluation config,
#         # such that evaluation always happens on the already updated policy,
#         # instead of on the already used train_batch.
#         #TODO: This may tot be needed
#         if "policy_main_reward" in result["hist_stats"]:
#             #print("HERE: ", result["hist_stats"])
#             # main_rew = result["hist_stats"].pop("policy_main_reward")
#             # #opp_rew = result["hist_stats"].pop("policy_gene_2_policy_reward")
#             # opponent_rew = list(result["hist_stats"].values())[0]
#             # print("COMBINED: ",sum(opponent_rew))
#             # assert len(main_rew) == len(opponent_rew)
#             # won = 0
#             # for r_main, r_opponent in zip(main_rew, opponent_rew):
#             #     if r_main > r_opponent:
#             #         won += 1

#             #print("HERE: ", result["sampler_results"]["policy_reward_min"].keys())
#             win_rate = result["custom_metrics"]["win_mean"]
#             result["win_rate"] = win_rate

#             print(f"Iter={algorithm.iteration} win-rate={win_rate} -> ", end="")

#             # If win rate is good -> Snapshot current policy and play against
#             # it next, keeping the snapshot fixed and only improving the "main"
#             # policy.
#             if win_rate > 0.55: # win_rate_treshold
#                 self.current_opponent += 1
#                 new_pol_id = f"main_v{self.current_opponent}"
#                 print(f"Adding new opponent to the mix ({new_pol_id}).")
#                 self.policy_names.add(new_pol_id)

#                 print("NAMES: ", self.policy_names)

#                 # Re-define the mapping function, such that "main" is forced
#                 # to play against any of the previously played policies
#                 # (excluding "random").
#                 def policy_mapping_fn(agent_id, episode, worker, **kwargs):
#                     return (
#                         "main"
#                         if agent_id.startswith("gene-1")
#                         else "main_v{}".format(
#                             np.random.randint(self.last_removed_opponent, self.current_opponent + 1)
#                         )
#                     )

#                 main_policy = algorithm.get_policy("main")
#                 # if algorithm.config._enable_learner_api:
#                 #     new_policy = algorithm.add_policy(
#                 #         policy_id=new_pol_id,
#                 #         policy_cls=type(main_policy),
#                 #         policy_mapping_fn=policy_mapping_fn,
#                 #         module_spec=SingleAgentRLModuleSpec.from_module(main_policy.model),
#                 #     )
#                 # else:
#                 new_policy = algorithm.add_policy(
#                     policy_id=new_pol_id,
#                     policy_cls=type(main_policy),
#                     policy_mapping_fn=policy_mapping_fn,
#                     policies_to_train=["main"],
#                 )

#                 # Set the weights of the new policy to the main policy.
#                 # We'll keep training the main policy, whereas `new_pol_id` will
#                 # remain fixed.
#                 main_state = main_policy.get_state()
#                 new_policy.set_state(main_state)

#                 # Too many policies makes distributed learning hard
#                 # due to communication size
#                 # if len(self.policy_names) > 3:

#                 #     # Remove policy from algorithm
#                 #     algorithm.remove_policy(policy_id="main_v{}".format(
#                 #             self.last_removed_opponent
#                 #         ))
                    
#                 #     # Remove policy from name list
#                 #     self.policy_names.remove("main_v{}".format(
#                 #             self.last_removed_opponent
#                 #         ))
                    
#                 #     print("REMOVING: ", "main_v{}".format(self.last_removed_opponent))
#                 #     self.last_removed_opponent += 1

#                 #     # Due to the nature of the env_runner in the rollout worker we have to update the environments at
#                 #     # episode end of the first trajectory in the next training iteration --- limitation with how
#                 #     # env_runner is configured though minor item
#                 #     def update_worker_callback_flag(worker):
#                 #         worker.callbacks.policy_names = worker.callbacks.policy_names
#                 #         worker.callbacks.update_fn = True
#                 #         worker.callbacks.last_removed_opponent += 1
#                 #         worker.set_policy_mapping_fn(policy_mapping_fn)
#                 #         print("HUH: ", worker.callbacks.last_removed_opponent)

#                 #     algorithm.workers.foreach_worker(update_worker_callback_flag)

#                 # We need to sync the just copied local weights (from main policy)
#                 # to all the remote workers as well.
#                 algorithm.workers.sync_weights()
#             else:
#                 print("not good enough; will keep learning ...")

#             # +2 = main + random
#             result["league_size"] = self.current_opponent + 2
#         else:
#             print("HERE:", result["hist_stats"])

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
    register_env(select_env, lambda config: Survival_Map(config))

    # Configure the environment and create agent
    # https://docs.ray.io/en/latest/rllib/rllib-training.html#specifying-environments
    config_dict = {
        "env": select_env,
        "render_env": False,
        #"clip_rewards": 0.2,
        "env_config": {
            "size": 30,
            "brains": [
                    DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(),
                    ],
            "max_agents": 40,
            "render_mode": "rgb_array"
        },
        "gamma": 0.99,
        "kl_coeff": 1.0,
        "num_workers": 1,
        #change back to 20
        #"num_envs_per_worker": 1,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        # These two may break stuff
        # https://github.com/ray-project/ray/issues/12709#issuecomment-741737472
        # https://chuacheowhuan.github.io/RLlib_trainer_config/
        "horizon": -1,#2000,
        "soft_horizon": -1,
        "use_critic": True,
        "use_gae": True,
        "shuffle_sequences": True,
        "vf_loss_coeff": 1.0,
        "vf_clip_param": 10.0,
        "kl_target": 0.01,
        # "clip_rewards": True,
        "clip_param": 0.25,
        # Number of SGD iterations in each outer loop
        # (i.e., number of epochs to execute per train batch).
        "num_sgd_iter": 24,
        # Divide episodes into fragments of this many steps each during rollouts.
        # Sample batches of this size are collected from rollout workers and
        # combined into a larger batch of `train_batch_size` for learning.
        "rollout_fragment_length": "auto",
        # Whether to rollout "complete_episodes" or "truncate_episodes" to
        # `rollout_fragment_length` length unrolls. Episode truncation guarantees
        # evenly sized batches, but increases variance as the reward-to-go will
        # need to be estimated at truncation boundaries.
        "batch_mode": "truncate_episodes",
        "sgd_minibatch_size": 1638,
        # Training batch size, if applicable. Should be >= rollout_fragment_length.
        # Samples batches will be concatenated together to a batch of this size,
        # which is then passed to SGD.
        "train_batch_size": 3686,
        "entropy_coeff": 0.001,
        "lr": 5e-04,
        "lambda": 0.76,
        "model": {
            "custom_model": "rnn",
            "max_seq_len": 20,
        },
        "framework": "torch",
        "multiagent": {
            # When you define the policy and map the agent_id to that policy, you
            # are defining the action and observation space for that agent.
            "policy_mapping_fn": policy_mapper,
            "policies": {"main", "gene_2_policy"},
            # Always just train the "main" policy.
            "policies_to_train": ["main", "gene_2_policy"], #"gene_2_policy"],
            "policy_map_cache": "/opt/project/temp_cache",
            "policy_map_capacity": 200
        },

        # "callbacks": SelfPlayCallback,
    }

    # algo = ppo.PPO(env=select_env, config=config_dict) 

    # status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    # n_iter = 1000

    train = False
    # if train == True:
    #     # train a policy with RLlib using PPO
    #     for n in range(n_iter):
    #         # Run the episodes
    #         result = algo.train()
    #         # Save a checkpoint of the latest policy
    #         chkpt_file = algo.save(chkpt_root)

    #         print(status.format(
    #                 n + 1,
    #                 result["episode_reward_min"],
    #                 result["episode_reward_mean"],
    #                 result["episode_reward_max"],
    #                 result["episode_len_mean"],
    #                 chkpt_file
    #                 ))


    #     # Examine one of the trained policies
    #     policy = algo.get_policy("main")
    #     model = policy.model
    #     print(model)

    if train == True:
        stopping_criteria = {"training_iteration": 200, "episode_reward_mean": 16000, "episode_len_mean": 3000}

        analysis = tune.run(
            "PPO",
            config=config_dict,
            stop=stopping_criteria,
            checkpoint_freq=5,
            checkpoint_at_end=True,
            checkpoint_score_attr="episode_reward_mean",
        )

        # restore a trainer from the last checkpoint
        trial = analysis.get_best_logdir("episode_reward_mean", "max")
        checkpoint = analysis.get_best_checkpoint(
            trial,
            "episode_reward_mean",
            "max",
        )

    else:
        def policy_mapper_test(agent_id, episode, worker):
            if agent_id.startswith("gene-1"):
                return "main"
            else:
                return "main"
        
        config_dict["multiagent"]["policy_mapping_fn"] = policy_mapper_test
        config_dict["callbacks"] = DefaultCallbacks

    config_dict["explore"] = False
    trainer = PPOTrainer(config=config_dict)
    trainer.restore("C:\\Users\\alexa\\ray_results\\PPO\\PPO_survival-map-v0_b76a4_00000_0_2023-05-16_10-12-01\\checkpoint_000133")#"C:\\Users\\alexa\\ray_results\\PPO\\PPO_survival-map-v0_045a9_00000_0_2023-05-01_15-46-55\\checkpoint_000115")#"C:\\Users\\alexa\\ray_results\\PPO\\PPO_survival-map-v0_1b8aa_00000_0_2023-05-01_08-45-14\\checkpoint_000100")#"C:\Users\alexa\ray_results\PPO\PPO_survival-map-v0_0d507_00000_0_2023-04-30_09-36-07\checkpoint_000122")

    # if train == False:
    #     config_dict["explore"] = False
    #     algo = ppo.PPO(env=select_env, config=config_dict)

    # Apply the trained policy in a rollout
    # algo.from_checkpoint(
    #     "C:\\imperial\\MengProject\\saved_models\\checkpoint_000047",
    #     # Tell the `from_checkpoint` util to create a new Algo, but only with "main" in it.
    #     policy_ids=["main"],
    #     # Make sure to update the mapping function (we must not map to "main_v..." anymore
    #     # to avoid a runtime error).
    #     policy_mapping_fn=lambda agent_id, episode, worker, **kw: "main",
    #     # Since we defined this above, we have to re-define it here with the updated
    #     # PolicyIDs, otherwise, RLlib will throw an error (it will think that there is an
    #     # unknown PolicyID in this list ("main_v...")).
    #     policies_to_train=["main"],
    # )
    #algo.restore(chkpt_file)

    lstm_cell_size = 32

    config_dict["env_config"]["render_mode"] = "human"
    env = Survival_Map(config_dict["env_config"])

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
            action[agent_id], state_dict[agent_id], _ = trainer.compute_single_action(agent_obs, state=state_dict[agent_id], policy_id=policy_mapper(agent_id, 1, 1))

        #time.sleep(0.2)
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