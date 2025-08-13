# -*- coding: utf-8 -*-
import sys
IN_COLAB = "google.colab" in sys.modules

# ---------- Imports ----------
# Gym / Gymnasium compatibility
try:
    import gym
except Exception:
    import gymnasium as gym  # type: ignore

import numpy as np
import tensorflow as tf

# Optional TFP (để sample bằng distribution object)
USE_TFP = False
if USE_TFP:
    import tensorflow_probability as tfp  # pip install tensorflow_probability==0.21.0

# Fix cho numpy>=2.0
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from collections import deque
import os

# ---------- Hyperparameters ----------
hidden_size   = 32
actor_lr      = 7e-3
critic_lr     = 7e-3
gamma         = 0.99
max_episodes  = 300
seed          = 2000
RENDER        = False          # True để xem cửa sổ render khi train
MA_WINDOW     = 20             # moving average window cho reward
PLOT_EVERY    = 5              # update biểu đồ mỗi N episode
SAVE_PLOTS    = True           # lưu hình loss/return

# ---------- Actor / Critic ----------
class Actor(tf.keras.Model):
    """Policy: xuất softmax probs cho discrete actions."""
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.l1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.l2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.pi = tf.keras.layers.Dense(action_size, activation='softmax')  # probs

    def call(self, state):
        x = self.l1(state)
        x = self.l2(x)
        return self.pi(x)  # (B, A)

class CriticV(tf.keras.Model):
    """Value network: V(s)."""
    def __init__(self, state_size: int):
        super().__init__()
        self.l1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.l2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.v  = tf.keras.layers.Dense(1, activation=None)

    def call(self, state):
        x = self.l1(state)
        x = self.l2(x)
        return self.v(x)  # (B,1)

# ---------- Agent (A2C) ----------
class A2CAgent:
    def __init__(self, env: gym.Env):
        self.env = env
        self.state_size  = int(self.env.observation_space.shape[0])
        self.action_size = int(self.env.action_space.n)

        self.actor  = Actor(self.state_size, self.action_size)
        self.critic = CriticV(self.state_size)

        self.a_opt = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        # Build weights sớm (tránh lazy-build)
        dummy = tf.zeros((1, self.state_size), dtype=tf.float32)
        _ = self.actor(dummy); _ = self.critic(dummy)

    @staticmethod
    def _to_batch(s):
        return np.array([s], dtype=np.float32)

    def get_action(self, state: np.ndarray) -> int:
        """Stochastic chọn hành động (dùng khi train)."""
        s = self._to_batch(state)
        probs = self.actor(s, training=False).numpy()  # (1, A)
        if USE_TFP:
            dist = tfp.distributions.Categorical(probs=probs, dtype=tf.float32)
            a = int(dist.sample().numpy()[0])
        else:
            a = int(np.random.choice(len(probs[0]), p=probs[0]))
        return a

    def get_action_deterministic(self, state: np.ndarray) -> int:
        """Chọn hành động argmax (dùng khi visualize để ổn định)."""
        s = self._to_batch(state)
        probs = self.actor(s, training=False).numpy()[0]  # (A,)
        return int(np.argmax(probs))

    def _log_prob_no_tfp(self, probs: tf.Tensor, action: int) -> tf.Tensor:
        """log π(a|s) khi không dùng TFP."""
        pa = tf.gather(probs[0], action)
        pa = tf.clip_by_value(pa, 1e-8, 1.0)
        return tf.math.log(pa)[tf.newaxis]  # (1,)

    def train_step(self, state, action, reward, next_state, done):
        s  = self._to_batch(state)
        s2 = self._to_batch(next_state)

        r  = tf.convert_to_tensor([[float(reward)]], dtype=tf.float32)        # (1,1)
        d  = tf.convert_to_tensor([[1.0 if done else 0.0]], dtype=tf.float32) # (1,1)

        with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:
            probs  = self.actor(s, training=True)         # (1,A)
            v_s    = self.critic(s, training=True)        # (1,1)
            v_s2   = tf.stop_gradient(self.critic(s2, training=True))

            target = r + (1.0 - d) * gamma * v_s2
            td     = target - v_s                         # (1,1)

            # ----- Critic loss -----
            critic_loss = tf.reduce_mean(tf.keras.losses.mse(target, v_s))

            # ----- Actor loss -----
            adv_detached = tf.stop_gradient(td)           # detach advantage
            if USE_TFP:
                dist = tfp.distributions.Categorical(probs=probs, dtype=tf.float32)
                logp = dist.log_prob([action])            # (1,)
            else:
                logp = self._log_prob_no_tfp(probs, action)  # (1,)
            actor_loss = - tf.reduce_mean(logp * tf.squeeze(adv_detached, axis=-1))

        g_a = tape_a.gradient(actor_loss, self.actor.trainable_variables)
        g_c = tape_c.gradient(critic_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(g_a, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(g_c, self.critic.trainable_variables))

        return float(actor_loss.numpy()), float(critic_loss.numpy())

# ---------- Make env (gym 0.26+ compatible) ----------
env_name = "CartPole-v1"
render_mode = "human" if RENDER else None
try:
    env = gym.make(env_name, render_mode=render_mode)
except TypeError:
    env = gym.make(env_name)

# Seeding API mới
np.random.seed(seed)
tf.random.set_seed(seed)
try:
    env.action_space.seed(seed)
    reset_out = env.reset(seed=seed)
    init_state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
except Exception:
    # gym cũ
    env.seed(seed)
    reset_out = env.reset()
    init_state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

agent = A2CAgent(env)

# ---------- Visualization (training curves) ----------
plt.ion()
fig1, ax1 = plt.subplots(figsize=(7,4))   # Reward curves
fig2, ax2 = plt.subplots(figsize=(7,4))   # Loss curves

def update_reward_plot(ep_returns, ma_window=MA_WINDOW):
    ax1.clear()
    ax1.plot(ep_returns, label="Return per episode")
    if len(ep_returns) >= 2:
        w = min(ma_window, len(ep_returns))
        ma = np.convolve(ep_returns, np.ones(w)/w, mode='valid')
        ax1.plot(range(w-1, w-1+len(ma)), ma, label=f"Moving Avg ({w})")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Return")
    ax1.grid(True); ax1.legend()
    fig1.tight_layout(); plt.pause(0.001)

def update_loss_plot(actor_losses, critic_losses):
    ax2.clear()
    ax2.plot(actor_losses, label="Actor loss")
    ax2.plot(critic_losses, label="Critic loss")
    ax2.set_xlabel("Update step (episodes)"); ax2.set_ylabel("Loss")
    ax2.grid(True); ax2.legend()
    fig2.tight_layout(); plt.pause(0.001)

# ---------- Animation (evaluation episode) ----------
def render_episode_animation(agent: A2CAgent,
                             env_name: str = "CartPole-v1",
                             fps: int = 30,
                             max_steps: int = 500,
                             save_path: str = "cartpole_animation.gif",
                             deterministic: bool = True,
                             seed: int = 42):
    """
    Chạy 1 episode với render_mode='rgb_array', tạo animation bằng matplotlib.
    - deterministic=True: dùng argmax policy để hình ổn định.
    - save_path: '.gif' (PillowWriter) hoặc '.mp4' (ffmpeg nếu có).
    """
    # Env cho render ảnh
    try:
        eval_env = gym.make(env_name, render_mode="rgb_array")
    except TypeError:
        # gym cũ có thể dùng mode="rgb_array"
        eval_env = gym.make(env_name)

    # Reset theo API mới
    try:
        obs, _ = eval_env.reset(seed=seed)
    except Exception:
        tmp = eval_env.reset()
        obs = tmp[0] if isinstance(tmp, tuple) else tmp

    frames = []
    state = obs
    done  = False
    t     = 0

    # Lấy frame đầu
    frame0 = eval_env.render()
    frames.append(frame0)

    while not done and t < max_steps:
        if deterministic:
            action = agent.get_action_deterministic(state)
        else:
            action = agent.get_action(state)

        step_out = eval_env.step(action)
        if len(step_out) == 5:
            next_state, reward, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
        else:
            next_state, reward, done, _ = step_out

        frame = eval_env.render()
        frames.append(frame)

        state = next_state
        t += 1

    eval_env.close()

    # Tạo animation bằng matplotlib
    fig, ax = plt.subplots(figsize=(5,4))
    ax.axis("off")
    im = ax.imshow(frames[0])

    def _update(i):
        im.set_data(frames[i])
        return (im,)

    anim = animation.FuncAnimation(fig, _update,
                                   frames=len(frames),
                                   interval=1000.0 / max(fps,1),
                                   blit=True)
    plt.tight_layout()
    plt.show(block=False)

    # Lưu file nếu có đường dẫn
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            if save_path.lower().endswith(".gif"):
                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer)
            elif save_path.lower().endswith(".mp4"):
                # Cần ffmpeg trong PATH; nếu không có, tự động fallback sang GIF tên khác
                try:
                    FFMpegWriter = animation.FFMpegWriter
                    writer = FFMpegWriter(fps=fps, bitrate=1800)
                    anim.save(save_path, writer=writer)
                except Exception:
                    alt = os.path.splitext(save_path)[0] + ".gif"
                    writer = PillowWriter(fps=fps)
                    anim.save(alt, writer=writer)
                    print(f"[Warn] ffmpeg không khả dụng, đã lưu GIF: {alt}")
            else:
                # Mặc định GIF
                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer)
            print(f"[OK] Saved animation to: {save_path}")
        except Exception as e:
            print(f"[Warn] Không lưu được animation: {e}")

    # Giữ cửa sổ animation mở khi chạy ngoài notebook
    if not IN_COLAB:
        plt.show()

# ---------- Train ----------
if __name__ == "__main__":
    scores = []
    actor_losses_hist  = []
    critic_losses_hist = []

    for ep in range(1, max_episodes + 1):
        reset_out = agent.env.reset()
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done = False
        ep_ret = 0.0

        while not done:
            action = agent.get_action(state)
            step_out = agent.env.step(action)

            # Gym 0.26+ trả 5 giá trị
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                next_state, reward, done, _ = step_out

            a_loss, c_loss = agent.train_step(state, action, reward, next_state, done)
            state = next_state
            ep_ret += float(reward)

        scores.append(ep_ret)
        actor_losses_hist.append(a_loss)
        critic_losses_hist.append(c_loss)

        # Console log
        print(f"Episode {ep:03d} | Return: {ep_ret:.1f} | a_loss={a_loss:.4f} | c_loss={c_loss:.4f}")
