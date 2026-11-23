from GridWorld import GridWorld
from agents.baseline import manhattan_distance
from agents.DQN import DQNAgent

SIZE = (10, 10)
START_POINT = (1, 2)
END_POINT = (5, 5)

env = GridWorld(size=SIZE)
env.init_env(start_point=START_POINT, end_point=END_POINT)
env.print_grid()

baseline = manhattan_distance(start_point=START_POINT, end_point=END_POINT)
print(f"Manhetten distance: {baseline}")

agent = DQNAgent()
agent.build_model()
