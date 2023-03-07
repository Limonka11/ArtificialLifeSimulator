from typing import List
import sys

sys.path.append("C:\\imperial\\MengProject\\Environment")
from environment import Environment
from Brain import Brain
from DQNBrain import DQNBrain
from A2CBrain import DDQNBrain

def simulator(brains: List[Brain],
            width: int,
            height: int,
            max_agents: int,
            fps: int):

    env = Environment(
        width=width,
        height=height,
        brains=brains,
        grid_size=10,
        max_agents=max_agents,
        training=False)

    episode_number = 0

    env.init_environment()
    env.render(fps=fps)

    while True:
        for agent in env.agents:
            agent.action = agent.brain.act(agent.state, episode_number)

        env.update_step()
        env.update_environment()
        env.render(fps=fps)

        episode_number += 1

main_brains = [DQNBrain(), DQNBrain(), DQNBrain(), DQNBrain(), DQNBrain(),
               DQNBrain(), DQNBrain(), DQNBrain(), DQNBrain(), DQNBrain(),
               DQNBrain(), DQNBrain(), DQNBrain(), DQNBrain(), DQNBrain(),
               DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(),
               DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(),
               DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain(), DDQNBrain()]

simulator(main_brains, width=50, height=50, max_agents=30, fps=10)