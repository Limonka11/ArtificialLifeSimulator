from typing import List
import sys
import atexit

sys.path.append("C:\\imperial\\MengProject\\Environment")
from environment import Environment
from Brain import Brain, RandBrain
from DQNBrain import DQNBrain
from A2CBrain import DDQNBrain

def simulator(brains: List[Brain],
            width: int,
            height: int,
            max_agents: int,
            fps: int,
            training: bool = False):

    env = Environment(
        width=width,
        height=height,
        brains=brains,
        square_size=15,
        max_agents=max_agents,
        training=training)

    episode_number = 0

    env.init_environment()
    env.render(fps=fps)

    atexit.register(env.show_stats)

    while True:
        for agent in env.agents:
            agent.action = agent.brain.act(agent.state, episode_number)

        env.update_step()

        # Learn based on the result
        for agent in env.agents:
            agent.learn(n_epi=episode_number)

        env.update_environment()
        env.render(fps=fps)

        episode_number += 1

if (__name__ == "__main__"):
#load_model="20230328-164750/model.pth"
    main_brains = [DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(),
                   DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(),
                   DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(),
                   DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain()]

    simulator(main_brains, width=50, height=50, max_agents=20, fps=10)

