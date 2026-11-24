from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from rich.console import Console
from rich.table import Table

from GridWorld import GridWorld
from draw_stats import draw_stats
from agents.baseline import manhattan_distance
from agents.DQN import DQNAgent

def train(
    size: tuple[int, int] = (10, 10),
    episodes: int = 5000,
    max_steps: int = 100,
    log_steps: int = 50,
    random_state: int = 42,
    agent_name: str = "dqn_v1",
    state_size: int = 4,
    action_size: int = 4,
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    memory_size: int = 10000,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, Path]:
    """Train an agent and return episode logs and the run directory."""
    if size[0] <= 0 or size[1] <= 0 or sum(size) < 4:
        raise ValueError("The grid must contain points at least two steps apart")
    if episodes <= 0 or max_steps <= 0 or log_steps <= 0:
        raise ValueError("episodes, max_steps and log_steps must be positive")
    if state_size != 4 or action_size != 4:
        raise ValueError("GridWorld requires 4 state values and 4 actions")

    tf.keras.utils.set_random_seed(random_state)
    tf.config.experimental.enable_op_determinism()

    rng = np.random.default_rng(random_state)
    env = GridWorld(size=size)
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        memory_size=memory_size,
        batch_size=batch_size,
        random_state=random_state,
    )
    agent.build_model(learning_rate=learning_rate)
    console = Console()
    episode_logs = []

    def print_log_row(*values: str, style: str = "") -> None:
        table = Table.grid(expand=True, padding=(0, 1))
        for _ in values:
            table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        table.add_row(*values, style=style)
        console.print(table)

    def random_point() -> tuple[int, int]:
        width, height = size
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        return x, y

    console.rule("Start Training", style="cyan")
    print_log_row("Episode", "Reward", "Steps", "Done", "Epsilon", "Efficiency", "Progress", style="bold cyan")
    for episode in range(episodes):
        start = random_point()
        end = random_point()

        while manhattan_distance(start, end) < 2:
            end = random_point()

        state = env.init_env(start_point=start, end_point=end)
        total_reward = 0.0
        done = False
        out_of_bounds = False

        for step in range(max_steps):
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
        distance = manhattan_distance(start_point=start, end_point=end)
        efficiency = distance / (step + 1) if done else None

        episode_logs.append({
            "Episode": episode + 1,
            "reward": total_reward,
            "steps": step + 1,
            "Done": done,
            "Epsilon": agent.epsilon,
            "Efficiency": efficiency,
        })

        if (episode + 1) % log_steps == 0:
            print_log_row(
                f"{episode + 1}/{episodes}",
                f"[{'green' if total_reward > 0 else 'red' if total_reward < 0 else 'white'}]{total_reward:.1f}[/]",
                str(step + 1),
                f"[{'green' if done else 'red'}]{done}[/]",
                f"{agent.epsilon:.3f}",
                f"{efficiency:.2f}" if efficiency is not None else "-",
                f"{(episode + 1) / episodes * 100:.1f}%",
            )

    logs = pd.DataFrame(episode_logs, columns=["Episode", "reward", "steps", "Done", "Epsilon", "Efficiency"])
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_dir = Path("agents/history") / agent_name / run_name
    save_dir.mkdir(parents=True, exist_ok=False)
    logs.to_pickle(save_dir / "logs.pkl")
    agent.save(str(save_dir / "model.weights.h5"))
    draw_stats(logs, save_dir)
    console.rule(f"Results saved in [italic]'{save_dir}'[/]", style="cyan")
    return logs, save_dir


if __name__ == "__main__":
    train()
