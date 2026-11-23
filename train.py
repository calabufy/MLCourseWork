import numpy as np

from GridWorld import GridWorld
from agents.baseline import manhattan_distance
from agents.DQN import DQNAgent

SIZE = (10, 10)
EPISODES = 5000
MAX_STEPS = 100
RANDOM_STATE = 42

rng = np.random.default_rng(RANDOM_STATE)
env = GridWorld(size=SIZE)
agent = DQNAgent(random_state=RANDOM_STATE)
agent.build_model()

def random_point() -> tuple[int, int]:
    width, height = SIZE
    x = int(rng.integers(0, width))
    y = int(rng.integers(0, height))
    return x, y

print("=== Start Training ===")
for episode in range(EPISODES):
    start = random_point()
    end = random_point()

    while end == start:
        end = random_point()

    state = env.init_env(start_point=start, end_point=end)
    total_reward = 0.0
    done = False
    out_of_bounds = False

    for step in range(MAX_STEPS):
        action = agent.choose_action(state)

        observation, reward, out_of_bounds, done = env.step(action)
        next_state = np.asarray(observation)

        agent.remember(
            state, action, reward, next_state,
            out_of_bounds, done
        )

        agent.train()

        state = next_state
        total_reward += reward

        if done or out_of_bounds:
            break

    agent.decay_epsilon()

    if (episode + 1) % 100 == 0:
        print(
            f"Episode: {episode + 1}"
            f"  reward: {total_reward:.1f}"
            f"  steps: {step + 1}"
            f"  done: {done}"
            f"  epsilon: {agent.epsilon:.3f}"
        )

save_path ="agents/history/dqn_v1.weights.h5" 
agent.save(save_path)
print(f"=== Agent saved in '{save_path}'===")