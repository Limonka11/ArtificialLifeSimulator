import random
import pprint
import os
import numpy as np

import ray
from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from gym_example.envs.Survival_Map import Survival_Map
from gym_example.envs.Entities.lstmBrain import TorchRNNModel
from gym_example.envs.Entities.A2CBrain import DDQNBrain

def policy_mapper(agent_id, episode, worker):
    if agent_id.startswith("gene-1"):
        return "main"
    else:
        return "gene_2_policy"

if __name__ == "__main__":
    import argparse

    # Register environment
    register_env("survival-map-v0", lambda config: Survival_Map(config))

    ModelCatalog.register_custom_model(
        "rnn", TorchRNNModel)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    args, _ = parser.parse_known_args()

    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

    hyperparam_mutations = {
        "lambda": [0.75, 0.80, 0.85, 0.90, 0.95],
        "clip_param": [0.15, 0.20, 0.25, 0.30, 0.35, 0.5],
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "gamma": [0.9, 0.93, 0.95, 0.97, 0.99],
        "entropy_coeff": [0, 1e-2, 5e-3, 1e-4, 5e-5],
        "num_sgd_iter": [1, 3, 5, 10, 15, 20, 25],
        "sgd_minibatch_size": lambda: random.randint(128, 16384),
        "train_batch_size": lambda: random.randint(200, 32000),
    }

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations=hyperparam_mutations,
        custom_explore_fn=explore,
    )

    # Stop when we've either reached 100 training iterations or reward=300
    stopping_criteria = {"training_iteration": 500, "episode_reward_mean": 10000}

    tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            scheduler=pbt,
            num_samples=1 if args.smoke_test else 4,
        ),
        param_space={
            "env": "survival-map-v0",
            "env_config": {
                "size": 40,
                "brains": [DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain()],
                "max_agents": 20,
                "render_mode": "rgb_array"
            },
            "kl_coeff": 1.0,
            "num_workers": 1,
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),  # number of GPUs to use per trial
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
                "policies_to_train": ["main", "gene_2_policy"],
                "policy_map_cache": "/opt/project/temp_cache",
                "policy_map_capacity": 200,
                "count_steps_by": "agent_steps"
            },
            "use_critic": True,
            "use_gae": True,
            "shuffle_sequences": True,
            "vf_loss_coeff": 1.0,
            "vf_clip_param": 10.0,
            "kl_target": 0.01,
            "batch_mode": "truncate_episodes",
            # These params are tuned from a fixed starting value.
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.2,
            "lr": 1e-3,
            "entropy_coeff": 0,
            "sgd_minibatch_size": 2048,
            # These params start off randomly drawn from a set.
            "num_sgd_iter": tune.choice([10, 15, 25]),
            "train_batch_size": 4096,
        },
        run_config=air.RunConfig(stop=stopping_criteria),
    )
    results = tuner.fit()


    best_result = results.get_best_result()

    print("Best performing trial's final set of hyperparameters:\n")
    pprint.pprint(
        {k: v for k, v in best_result.config.items() if k in hyperparam_mutations}
    )

    print("\nBest performing trial's final reported metrics:\n")

    metrics_to_print = [
        "episode_reward_mean",
        "episode_reward_max",
        "episode_reward_min",
        "episode_len_mean",
    ]
    pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})