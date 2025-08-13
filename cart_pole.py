# ====== IMPORT LIBRARIES ======
import sys
import random
import gym
import numpy as np

# Bổ sung cho compatibility một số phiên bản numpy
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from collections import deque
import matplotlib.pyplot as plt

# ===== Q-Network - Mạng Neural Network học Q-value =====
class QNetwork(Model):
    def __init__(self, num_state_features: int, num_actions: int):
        super(QNetwork, self).__init__()
        # Lớp ẩn thứ 1 và 2 (kích thước đặt ở ngoài)
        self.hidden_layer1 = tf.keras.layers.Dense(hidden_units, activation='relu')
        self.hidden_layer2 = tf.keras.layers.Dense(hidden_units, activation='relu')
        # Lớp đầu ra - số node = số action, không có activation (linear)
        self.output_layer = tf.keras.layers.Dense(num_actions)
    def call(self, state_tensor):
        x = self.hidden_layer1(state_tensor)
        x = self.hidden_layer2(x)
        q_values = self.output_layer(x)
        return q_values

# ===== Agent DQN (Học & quyết định hành động) =====
class DQNAgent:
    def __init__(self, env: gym.Env, batch_size: int):
        self.env = env
        # Số chiều trạng thái (số đặc trưng của state)
        self.num_state_features = self.env.observation_space.shape[0]
        # Số lượng hành động có thể thực hiện
        self.num_actions = self.env.action_space.n
        self.batch_size = batch_size
        self.learning_rate = 0.001
        self.discount_factor = 0.99  # gamma

        # Mạng Q để ước lượng Q-value
        self.q_network = QNetwork(self.num_state_features, self.num_actions)
        # Optimizer cho việc học trọng số mạng
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        # Bộ nhớ kinh nghiệm dạng deque (buffer)
        self.replay_buffer = deque(maxlen=2000)

    # Chọn action theo epsilon-greedy
    def select_action(self, state, epsilon):
        # Dự đoán Q-value của tất cả action ở state hiện tại
        q_values = self.q_network(tf.convert_to_tensor([state], dtype=tf.float32))[0]
        # Quyết định khám phá hay khai thác
        if np.random.rand() <= epsilon:
            action = np.random.choice(self.num_actions)  # Chọn ngẫu nhiên (explore)
        else:
            action = np.argmax(q_values)                 # Chọn action Q-value lớn nhất (exploit)
        return action

    # Lưu một transition (state, action, reward, next_state, done) vào bộ nhớ
    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    # Một lần huấn luyện từ sample minibatch của replay buffer
    def train_from_replay(self):
        mini_batch = random.sample(self.replay_buffer, self.batch_size)
        batch_states      = [sample[0] for sample in mini_batch]
        batch_actions     = [sample[1] for sample in mini_batch]
        batch_rewards     = [sample[2] for sample in mini_batch]
        batch_next_states = [sample[3] for sample in mini_batch]
        batch_dones       = [sample[4] for sample in mini_batch]

        variables = self.q_network.trainable_variables
        with tf.GradientTape() as tape:
            # Chuyển batch sang tensor để tính toán với mạng nơ-ron
            states_tensor      = tf.convert_to_tensor(np.vstack(batch_states), dtype=tf.float32)
            actions_tensor     = tf.convert_to_tensor(batch_actions, dtype=tf.int32)
            rewards_tensor     = tf.convert_to_tensor(batch_rewards, dtype=tf.float32)
            next_states_tensor = tf.convert_to_tensor(np.vstack(batch_next_states), dtype=tf.float32)
            dones_tensor       = tf.convert_to_tensor(batch_dones, dtype=tf.float32)
            # Q-values cho state hiện tại
            current_Qs = self.q_network(states_tensor)
            # Q-value của action đã thực hiện (chọn ra đúng giá trị trong vector Q)
            chosen_Q_values = tf.reduce_sum(
                tf.one_hot(actions_tensor, self.num_actions) * current_Qs, axis=1
            )
            # Q-value của state kế tiếp (dùng chính q_network, nếu muốn chuẩn thì nên có target_network)
            next_Qs = self.q_network(next_states_tensor)
            best_next_actions = tf.argmax(next_Qs, axis=1)
            best_next_Q_values = tf.reduce_sum(
                tf.one_hot(best_next_actions, self.num_actions) * next_Qs, axis=1
            )
            # Nếu done=True thì không có giá trị kỳ vọng ở future (mask)
            mask = 1 - dones_tensor
            # Tính target: Q_target = reward + gamma * max_a Q(s', a) * mask
            target_Q_values = rewards_tensor + self.discount_factor * best_next_Q_values * mask
            # Hàm mất mát: MSE giữa Q thực tế và Q dự đoán
            error = tf.square(chosen_Q_values - target_Q_values)*0.5
            loss = tf.reduce_mean(error)
        # Lan truyền ngược, cập nhật trọng số
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

# ===== Hàm lưu biểu đồ học (reward theo tập) =====
def save_learning_curve(episode_rewards, episode_numbers):
    plt.figure(figsize=(8,4))
    plt.plot(episode_numbers, episode_rewards, label="Episode Reward")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training Curve')
    plt.legend()
    plt.grid()
    plt.savefig('reward_curve.png')
    plt.close()
    print("Reward curve saved as reward_curve.png")

# ===== MAIN: Huấn luyện agent với môi trường =====
env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="human")
hidden_units = 128
max_episodes = 200        # Số lần chơi
batch_size = 64

epsilon = 1.0                  # Khởi đầu tỉ lệ khám phá
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005

agent = DQNAgent(env, batch_size)

episode_rewards = []
episode_numbers = []
VISUALIZE_INTERVAL = 20

for episode in range(max_episodes):
    obs = agent.env.reset() # Đặt lại trạng thái môi trường
    if isinstance(obs, tuple):
        state, _ = obs      # Một số gym version trả tuple (obs, info)
    else:
        state = obs
    total_reward = 0
    done = False

    while not done:
        # Chọn action theo epsilon-greedy
        action = agent.select_action(state, epsilon)
        # Thực hiện action trong môi trường, nhận kết quả
        output = agent.env.step(action)
        if len(output) == 5:
            next_state, reward, terminated, truncated, info = output
            done = terminated or truncated
        else:
            next_state, reward, done, info = output
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        # Lưu transition vào buffer
        agent.store_experience(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        # Khi đủ dữ liệu, mới train một bước
        if len(agent.replay_buffer) >= agent.batch_size:
            agent.train_from_replay()
        # (Tùy chọn) render để quan sát quá trình học
        env.render()
    episode_rewards.append(total_reward)
    episode_numbers.append(episode + 1)
    print(f"Episode {episode+1}: Reward = {total_reward}")

    # Giảm dần epsilon theo thời gian (epsilon decay)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    # Lưu biểu đồ reward định kỳ
    if (episode + 1) % VISUALIZE_INTERVAL == 0 or episode == 0:
        save_learning_curve(episode_rewards, episode_numbers)

print("Training finished.")
save_learning_curve(episode_rewards, episode_numbers)
env.close()
