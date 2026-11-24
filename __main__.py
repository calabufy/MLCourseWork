from train import train

if __name__ == "__main__":
    # episodes experement
    for i in range(2500, 5000, 7500):
        train(episodes=i, log_steps=100)