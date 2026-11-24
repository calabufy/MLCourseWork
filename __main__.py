from train import train

if __name__ == "__main__":
    # episodes
    for episodes in [2500, 5000, 7500]:
        print(f"Experiment 1: episodes = {episodes}")
        train(episodes=episodes, log_steps=episodes//100, agent_name=f"DQN_episode{episodes}")

    # architecture
    for layers, neurons in [(1, 32), (2, 64), (3, 128)]:
        print(f"Experiment 2: architecture = {(layers, neurons)}")
        train(NN_layers=layers, NN_neurons=neurons, agent_name=f"DQN_architecture{layers}")

    # Exploration
    for epsilon_decay in [0.95, 0.995, 0.999]:
        print(f"Experiment 3: epsilon_decay = {epsilon_decay}")
        train(epsilon_decay=epsilon_decay, agent_name=f"DQN_decay{epsilon_decay}")