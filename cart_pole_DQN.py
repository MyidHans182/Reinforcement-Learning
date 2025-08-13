# IMPORTING LIBRARIES
import sys
IN_COLAB = "google.colab" in sys.modules

import random
import gym
import numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from collections import deque

# Fix cho numpy>=2.0 (nếu cần)
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# ---------------- Q-Network ----------------
class Network(Model):
    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        super().__init__()
        self.layer1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.layer2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.value  = tf.keras.layers.Dense(action_size)

    def call(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        return self.value(x)

class DQNAgent:
    def __init__(self, env: gym.Env, batch_size: int, target_update: int,
                 hidden_size: int = 128, lr: float = 1e-3, gamma: float = 0.99):
        self.env = env
        self.state_size  = int(self.env.observation_space.shape[0])
        self.action_size = int(self.env.action_space.n)

        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma

        self.dqn        = Network(self.state_size, self.action_size, hidden_size)
        self.dqn_target = Network(self.state_size, self.action_size, hidden_size)
        self.optimizer  = optimizers.Adam(learning_rate=lr)

        self.memory = deque(maxlen=20000)
        self._target_hard_update()

    def get_action(self, state, epsilon: float):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.action_size)
        q_value = self.dqn(tf.convert_to_tensor([state], dtype=tf.float32))[0]
        return int(tf.argmax(q_value).numpy())

    def append_sample(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def train_step(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        states      = tf.convert_to_tensor(np.vstack([b[0] for b in mini_batch]), dtype=tf.float32)
        actions     = tf.convert_to_tensor([b[1] for b in mini_batch], dtype=tf.int32)
        rewards     = tf.convert_to_tensor([b[2] for b in mini_batch], dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.vstack([b[3] for b in mini_batch]), dtype=tf.float32)
        dones       = tf.convert_to_tensor([b[4] for b in mini_batch], dtype=tf.float32)

        with tf.GradientTape() as tape:
            # Q(s,a)
            q_all = self.dqn(states)                                   # (B, A)
            q_sa  = tf.reduce_sum(tf.one_hot(actions, self.action_size) * q_all, axis=1)

            # Double-DQN target: argmax bằng mạng chính, giá trị bằng mạng target
            next_q_main  = self.dqn(next_states)
            next_actions = tf.argmax(next_q_main, axis=1)
            next_q_targ  = self.dqn_target(next_states)
            next_q_max   = tf.reduce_sum(tf.one_hot(next_actions, self.action_size) * next_q_targ, axis=1)

            target = rewards + (1.0 - dones) * self.gamma * next_q_max
            loss = tf.reduce_mean(0.5 * tf.square(q_sa - target))

        grads = tape.gradient(loss, self.dqn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.dqn.trainable_variables))
        return float(loss.numpy())

    def _target_hard_update(self):
        self.dqn_target.set_weights(self.dqn.get_weights())

# ---------------- ENV & TRAIN ----------------
env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="human")  # hoặc None nếu không cần hiển thị

# params
hidden_size  = 128
max_episodes = 200
batch_size   = 64
target_update = 100

# epsilon
epsilon = 1.0
max_epsilon, min_epsilon, decay_rate = 1.0, 0.01, 0.005

agent = DQNAgent(env, batch_size=batch_size, target_update=target_update, hidden_size=hidden_size)

if __name__ == "__main__":
    update_cnt = 0
    scores = []
    for episode in range(max_episodes):
        obs = agent.env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.get_action(state, epsilon)

            step_out = agent.env.step(action)
            # HỖ TRỢ CẢ 2 API:
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                next_state, reward, done, _ = step_out

            agent.append_sample(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward

            if len(agent.memory) >= agent.batch_size:
                agent.train_step()
                update_cnt += 1
                if update_cnt % agent.target_update == 0:
                    agent._target_hard_update()

        scores.append(episode_reward)
        print(f"Episode {episode+1}: {episode_reward}")
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
