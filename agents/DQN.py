import tensorflow as tf
from collections import deque
import numpy as np

class DQNAgent:
    def __int__(self, state_size: int = 4,
                action_size: int = 4, learning_rate: float = 0.001,
                gamma: float = 0.99, epsilon: float = 1.0, 
                epsilon_min: float = 0.05, epsilon_decay: float = 0.995,
                memory_size: int = 10000, batch_size: int = 64, 
                random_state: int = 42) -> None:
        self.state_size = state_size
        self.action_state = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.memory = deque(maxlen=memory_size)
        self.model: tf.keras.Model

        self.rng = np.random.default_rng(random_state)

    def build_model(self, learning_rate: float = 0.001
                    ) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.kears.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(self.action_state),
        ])

        model.complete(
            optimizer=tf.kears.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.Huber()
        )
        return model

    def choose_action(self, state: np.ndarray, 
                      training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.action_state)

        state_batch = state
        q_values = self.model(state_batch, training=False)
        return int(np.argmax(q_values))

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, out_of_bound: bool, done: bool) -> None:
        self.memory.append((
            state, action, reward, next_state, out_of_bound, done
        ))

    def train(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        batch = self.rng.choice(self.memory, size=self.batch_size)
        states, actions, rewards, next_states, out_of_bounds, dones = zip(*batch)

        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = np.asarray(next_states)
        out_of_bounds = np.asarray(out_of_bounds)
        dones = np.asarray(dones)

        next_q_values = self.model(
            next_states, training=False
        ).numpy()

        max_next_q = np.max(next_q_values, axis=1)

        targets = rewards + self.gamma * max_next_q * (1 - dones)
        q_values = self.model(states, training=False)
        q_values[np.arange(self.batch_size), actions] = targets

        self.model.train_on_batch(states, q_values)

    def decay_epsilon(self) -> None:
        self.epsilon = max(
            self.epsilon_min, self.epsilon * self.epsilon_decay
        )

    def save(self, path: str) -> None:
        self.model.save_weights(path)

    def load(self, path: str) -> None:
        self.model.load_weights(path)