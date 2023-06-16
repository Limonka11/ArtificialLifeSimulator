import random
import pprint
import os
import numpy as np
import argparse

import ray
from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from gym_example.envs.Survival_Map import Survival_Map
from gym_example.envs.Entities.lstmBrain import TorchRNNModel

parser = argparse.ArgumentParser()

parser.add_argument('--lambda-range',
                    nargs='+',
                    default=[0.75, 0.80, 0.85, 0.90, 0.95],
                    help="The range of values to be explored for lambda.")

parser.add_argument('--clip-param-range',
                    nargs='+',
                    default=[0.15, 0.20, 0.25, 0.30, 0.35, 0.5],
                    help="The range of values to be explored for the clip parameter.")

parser.add_argument('--lr-range',
                    nargs='+',
                    default=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
                    help="The range of values to be explored for the learning rate.")

parser.add_argument('--gamma-range',
                    nargs='+',
                    default=[0.9, 0.93, 0.95, 0.97, 0.99],
                    help="The range of values to be explored for gamma. Gamma is the discout rate.")

parser.add_argument('--entropy-coeff-range',
                    nargs='+',
                    default=[0, 1e-2, 5e-3, 1e-4, 5e-5],
                    help="The range of values to be explored for the entropy_coeff. This parameter \
                        speciefies how much the agent will be exploring.")


parser.add_argument('--num-sgd-iter-range',
                    nargs='+',
                    default=[1, 3, 5, 10, 15, 20, 25],
                    help="The range of values to be explored for number of sgd iteration.\
                        The number of epochs to execute per train batch")

parser.add_argument('--sgd-minibatch-size',
                    nargs=2,
                    default=(128, 16384),
                    metavar=("min", "max"),
                    help="Set the minimum and maximum value for the sgd_minibatch_size.",)

parser.add_argument('--train-batch-size',
                    nargs=2,
                    default=(200, 32000),
                    metavar=("min", "max"),
                    help="Set the minimum and maximum value for the train_batch_size.",)

parser.add_argument(
    "--stop-iters", type=int, default=1000, help="Number of iterations to tune."
)

parser.add_argument(
    "--stop-reward", type=float, default=5000, help="Reward at which we stop the process which achieved this reward."
)

parser.add_argument(
    "--perturb-interval", type=float, default=120, help="Models will be considered for perturbation at this interval."
)

parser.add_argument(
    "--resample-prob", type=float, default=0.25, help="The probability of resampling from theoriginal distribution."
)

parser.add_argument(
    "--num-samples", type=int, default=4, help="Number of times to sample from the hyperparameter space.\
        This is also the number of processes to spawn."
)

parser.add_argument(
    "--env-size",
    type=int,
    default=40,
    help="The size of the environment i.e. Size X Size.",
)

parser.add_argument(
    "--agent-count", type=int, default=30, help="The number of agents to be spawn in the beginning of each episode."
)

parser.add_argument(
    "--max-agent-count", type=int, default=35, help="The max number of agents that could exist at the same time."
)

parser.add_argument(
    "--prey-pred-ratio",
    type=int,
    default=30,
    help="The ratio which will be used for adding the different types of agents. For example, if \
    the --agent-count is 10 and --prey-pred-ratio is 2, then 5 rabbits and 5 wolves will be spawned."
)

parser.add_argument(
    "--food-prob",
    type=float,
    default=0.1,
    help="The probability to spawn a food entity in each empty grid cell at the beginning of each episode."
)

parser.add_argument(
    "--poison-prob",
    type=float,
    default=0.05,
    help="The probability to spawn a poison entity in each empty grid cell at the beginning of each episode."
)

parser.add_argument(
    "--tree-prob",
    type=float,
    default=0.02,
    help="The probability to spawn a tree entity in each empty grid cell at the beginning of each episode."
)

parser.add_argument(
    "--num-workers", type=int, default=1, help="The number of workers to spawn. Each worker will be running environments on it."
)

parser.add_argument(
    "--num-envs-per-worker",
    type=int,
    default=1,
    help="Number of environments to evaluate vector-wise per worker. This enables model inference batching, which can improve \
        performance for inference bottlenecked workloads."
)

parser.add_argument(
    "--num-gpus", type=int, default=0, help="The number of GPUs to use."
)

def policy_mapper(agent_id, episode, worker):
    if agent_id.startswith("gene-1"):
        return "main"
    else:
        return "gene_2_policy"

if __name__ == "__main__":
    # Register environment
    register_env("survival-map-v0", lambda config: Survival_Map(config))

    ModelCatalog.register_custom_model(
        "rnn", TorchRNNModel)

    args = parser.parse_args()

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
        "lambda": args.lambda_range,
        "clip_param": args.clip_param_range,
        "lr": args.lr_range,
        "gamma": args.gamma_range,
        "entropy_coeff": args.entropy_coeff_range,
        "num_sgd_iter": args.num_sgd_iter_range,
        "sgd_minibatch_size": lambda: random.randint(args.sgd_minibatch_size[0], args.sgd_minibatch_size[1]),
        "train_batch_size": lambda: random.randint(args.train_batch_size[0], args.train_batch_size[1]),
    }

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=args.perturb_interval,
        resample_probability=args.resample_prob,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations=hyperparam_mutations,
        custom_explore_fn=explore,
    )

    stopping_criteria = {"training_iteration": args.stop_iters,
                        "episode_reward_mean": args.stop_reward}

    sgd_minibatch_size_start = random.randint(args.sgd_minibatch_size[0], args.sgd_minibatch_size[1])
    tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            scheduler=pbt,
            num_samples=args.num_samples,
        ),
        param_space={
            "env": "survival-map-v0",
            "env_config": {
                "size": args.env_size,
                "brains": args.agent_count,
                "max_agents": args.max_agent_count,
                "prey_pred_ratio": args.prey_pred_ratio,
                "food_prob": args.food_prob,
                "poison_prob": args.poison_prob,
                "tree_prob": args.tree_prob,
                "render_mode": "rgb_array"
            },
            "kl_coeff": 1.0,
            "num_workers": args.num_workers,
            "num_envs_per_worker": args.num_envs_per_worker,
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", args.num_gpus)),  # number of GPUs to use per trial
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

            # These params start off randomly drawn from a set.
            "gamma": random.choice(args.gamma_range),
            "lambda": random.choice(args.lambda_range),
            "clip_param": random.choice(args.clip_param_range),
            "lr": random.choice(args.lr_range),
            "entropy_coeff": random.choice(args.entropy_coeff_range),
            "num_sgd_iter": random.choice(args.num_sgd_iter_range),
            "sgd_minibatch_size": sgd_minibatch_size_start,
            "train_batch_size": random.randint(2 * sgd_minibatch_size_start, args.train_batch_size[1]),
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