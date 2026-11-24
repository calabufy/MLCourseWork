from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def draw_stats(logs: pd.DataFrame, save_dir: str | Path) -> None:
    """Save episode statistics in the training run directory."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, layout="constrained")

    axes[0].plot(logs["Episode"], logs["steps"], color="tab:blue", linewidth=0.8)
    axes[0].set_ylabel("Steps")
    axes[0].set_title("Steps per episode")

    axes[1].plot(
        logs["Episode"], logs["reward"], color="tab:orange",
        linewidth=0.8, alpha=0.45, label="Reward",
    )
    reward_trend = logs["reward"].rolling(window=100, min_periods=1).mean()
    axes[1].plot(
        logs["Episode"], reward_trend, color="tab:red", linewidth=2,
        label="Moving average (100 episodes)",
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1, label="Zero")
    axes[1].set_ylabel("Reward")
    axes[1].set_title("Reward per episode")
    axes[1].legend()

    successful = logs.dropna(subset=["Efficiency"])
    axes[2].scatter(
        successful["Episode"], successful["Efficiency"],
        color="tab:green", s=10, alpha=0.4, label="Efficiency",
    )
    trend = successful["Efficiency"].rolling(window=100, min_periods=1).mean()
    axes[2].plot(
        successful["Episode"], trend, color="tab:red", linewidth=2,
        label="Moving average",
    )
    axes[2].set_ylabel("Efficiency")
    axes[2].set_xlabel("Episode")
    axes[2].set_title("Efficiency per successful episode")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.savefig(Path(save_dir) / "stats.png", dpi=150)
    plt.close(fig)
