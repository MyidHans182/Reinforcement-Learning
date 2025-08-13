# -*- coding: utf-8 -*-
import sys
IN_COLAB = "google.colab" in sys.modules  # Kiểm tra xem đang chạy trên Google Colab hay không

# Gym / Gymnasium compatibility
try:
    import gym
except Exception:
    import gymnasium as gym  # Nếu gym cũ không có thì import gymnasium

import numpy as np
import tensorflow as tf

# numpy>=2.0 compat (fix lỗi numpy.bool8 bị bỏ)
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# ---------------- Hyperparams ----------------
hidden_size   = 32      # Số neuron ở mỗi hidden layer
actor_lr      = 7e-3    # Learning rate cho Actor
critic_lr     = 7e-3    # Learning rate cho Critic
gamma         = 0.99    # Hệ số chiết khấu (discount factor)
max_episodes  = 100      # Số episode train
seed          = 2000    # Seed để tái lập kết quả

# --------- Tracing / Rendering ----------
RENDER            = True       # Bật/tắt animation hiển thị
RENDER_EVERY      = 1          # Tần suất render (mỗi bao nhiêu episode)
TRACE             = True       # Bật chế độ in thông tin debug
TRACE_EPISODES    = {1}         # In debug cho những episode này
TRACE_STEPS_LIMIT = 200         # Giới hạn số step debug trong 1 episode

np.set_printoptions(precision=4, suppress=True)  # Cấu hình in số đẹp

# ----------------- Utilities -----------------
def l2_norm(tensors):
    """Tính chuẩn L2 của danh sách tensor."""
    if tensors is None: return 0.0
    sq = 0.0
    for t in tensors:
        if t is None: 
            continue
        x = tf.reshape(t, [-1])
        sq += tf.reduce_sum(x * x)
    return float(tf.sqrt(sq + 1e-12).numpy())

def dense_manual_forward(x, layer, act=None):
    """
    Thực hiện tính toán forward thủ công cho 1 layer Dense:
    z = xW + b, sau đó áp dụng activation nếu có.
    """
    W, b = layer.get_weights()
    z = tf.matmul(x, tf.convert_to_tensor(W, dtype=x.dtype)) + tf.convert_to_tensor(b, dtype=x.dtype)
    if act is None:
        h = z
    elif act == "relu":
        h = tf.nn.relu(z)
    elif act == "softmax":
        h = tf.nn.softmax(z)
    else:
        raise ValueError("Unsupported activation in debug forward")
    return z, h

# ---------------- Networks ----------------
class Actor(tf.keras.Model):
    """Mạng Actor: đầu ra là xác suất chọn action"""
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.l1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.l2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.pi = tf.keras.layers.Dense(action_size, activation='softmax')  # Softmax để ra probs
    def call(self, state):
        x = self.l1(state)
        x = self.l2(x)
        return self.pi(x)

class CriticV(tf.keras.Model):
    """Mạng Critic: đầu ra là giá trị V(s)"""
    def __init__(self, state_size: int):
        super().__init__()
        self.l1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.l2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.v  = tf.keras.layers.Dense(1, activation=None)  # Giá trị V(s)
    def call(self, state):
        x = self.l1(state)
        x = self.l2(x)
        return self.v(x)

class A2CAgent:
    """Agent sử dụng Actor-Critic (A2C)"""
    def __init__(self, env: gym.Env):
        self.env = env
        self.state_size  = int(env.observation_space.shape[0])  # Số chiều state
        self.action_size = int(env.action_space.n)              # Số action

        # Tạo mạng
        self.actor  = Actor(self.state_size, self.action_size)
        self.critic = CriticV(self.state_size)

        # Optimizer
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        # Build model (dummy input)
        dummy = tf.zeros((1, self.state_size), dtype=tf.float32)
        _ = self.actor(dummy)
        _ = self.critic(dummy)

    @staticmethod
    def _b(s): 
        """Chuyển state thành batch (1,state_size)"""
        return np.array([s], dtype=np.float32)

    # Debug forward cho Actor
    def debug_forward_actor(self, s1x):
        """Trả về giá trị z/h từng layer của Actor"""
        z1, h1 = dense_manual_forward(s1x, self.actor.l1, act="relu")
        z2, h2 = dense_manual_forward(h1,  self.actor.l2, act="relu")
        z3, p  = dense_manual_forward(h2,  self.actor.pi, act="softmax")
        return {"z1": z1.numpy(), "h1": h1.numpy(),
                "z2": z2.numpy(), "h2": h2.numpy(),
                "logits": z3.numpy(), "probs": p.numpy()}

    # Debug forward cho Critic
    def debug_forward_critic(self, s1x):
        """Trả về giá trị z/h từng layer của Critic"""
        z1, h1 = dense_manual_forward(s1x, self.critic.l1, act="relu")
        z2, h2 = dense_manual_forward(h1,  self.critic.l2, act="relu")
        z3, v  = dense_manual_forward(h2,  self.critic.v,  act=None)
        return {"z1": z1.numpy(), "h1": h1.numpy(),
                "z2": z2.numpy(), "h2": h2.numpy(),
                "v_preact": z3.numpy(), "v": v.numpy()}

    def get_action(self, state) -> int:
        """Chọn action dựa trên xác suất từ Actor"""
        probs = self.actor(self._b(state), training=False).numpy()[0]
        return int(np.random.choice(len(probs), p=probs))

    def train_step(self, state, action, reward, next_state, done, want_trace=False):
        """
        Một bước train:
        - Cập nhật Actor theo policy gradient
        - Cập nhật Critic theo TD error
        - Lưu thông tin trace nếu cần
        """
        s  = self._b(state)
        s2 = self._b(next_state)
        r  = tf.convert_to_tensor([[float(reward)]], dtype=tf.float32)
        d  = tf.convert_to_tensor([[1.0 if done else 0.0]], dtype=tf.float32)

        # Lưu trọng số trước update (để đo độ thay đổi)
        a_before = [w.numpy().copy() for w in self.actor.trainable_variables]
        c_before = [w.numpy().copy() for w in self.critic.trainable_variables]

        with tf.GradientTape() as ta, tf.GradientTape() as tc:
            # Forward
            probs = self.actor(s, training=True)
            v_s   = self.critic(s, training=True)
            v_s2  = tf.stop_gradient(self.critic(s2, training=True))  # Không backprop qua next state

            # Tính TD target & TD error
            target = r + (1.0 - d) * gamma * v_s2
            td     = target - v_s

            # Critic loss = MSE(target, V(s))
            c_loss = tf.reduce_mean(tf.keras.losses.mse(target, v_s))

            # Actor loss = -logπ(a|s) * advantage
            adv = tf.stop_gradient(td)
            pa  = tf.gather(probs[0], action)
            pa  = tf.clip_by_value(pa, 1e-8, 1.0)  # Tránh log(0)
            logp = tf.math.log(pa)[tf.newaxis]
            a_loss = - tf.reduce_mean(logp * tf.squeeze(adv, axis=-1))

        # Tính gradient & update
        ga = ta.gradient(a_loss, self.actor.trainable_variables)
        gc = tc.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(ga, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(gc, self.critic.trainable_variables))

        # Đo độ thay đổi tham số (chuẩn L2)
        a_delta_norm = l2_norm([w - wb for w, wb in zip(self.actor.trainable_variables, a_before)])
        c_delta_norm = l2_norm([w - wb for w, wb in zip(self.critic.trainable_variables, c_before)])

        # Lưu thông tin trace để debug
        trace = None
        if want_trace:
            trace = {
                "state": np.array(s, dtype=np.float32),
                "actor": self.debug_forward_actor(s),
                "critic_s": self.debug_forward_critic(s),
                "critic_s2": self.debug_forward_critic(s2),
                "action": int(action),
                "reward": float(reward),
                "done": bool(done),
                "target": float(target.numpy().squeeze()),
                "td": float(td.numpy().squeeze()),
                "actor_loss": float(a_loss.numpy()),
                "critic_loss": float(c_loss.numpy()),
                "grad_actor_L2": l2_norm(ga),
                "grad_critic_L2": l2_norm(gc),
                "param_update_actor_L2": a_delta_norm,
                "param_update_critic_L2": c_delta_norm,
            }

        return float(a_loss.numpy()), float(c_loss.numpy()), trace

# --------- Khởi tạo môi trường ----------
env_name = "CartPole-v1"
render_mode = "human" if RENDER else None
try:
    env = gym.make(env_name, render_mode=render_mode)
except TypeError:
    env = gym.make(env_name)

# Seed để tái lập kết quả
np.random.seed(seed)
tf.random.set_seed(seed)
try:
    env.action_space.seed(seed)
except Exception:
    pass

# Reset môi trường
reset_out = env.reset(seed=seed)
_ = reset_out[0] if isinstance(reset_out, tuple) else reset_out

agent = A2CAgent(env)

# ---------------- Train loop ----------------
if __name__ == "__main__":
    scores = []
    for ep in range(1, max_episodes + 1):
        reset_out = agent.env.reset(seed=seed + ep)
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done = False
        ep_ret = 0.0
        step_id = 0

        print(f"\n===== Episode {ep} =====")
        while not done:
            step_id += 1

            # Debug Actor trước khi chọn action
            s1x = agent._b(state)
            a_dbg = agent.debug_forward_actor(s1x)
            probs = a_dbg["probs"][0]
            action = int(np.random.choice(len(probs), p=probs))

            # Step môi trường
            step_out = agent.env.step(action)
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                next_state, reward, done, _ = step_out

            # Train & trace
            want_trace = TRACE and (ep in TRACE_EPISODES) and (step_id <= TRACE_STEPS_LIMIT)
            aloss, closs, t = agent.train_step(state, action, reward, next_state, done, want_trace)

            # In debug nếu cần
            if want_trace:
                print(f"\n--- Step {step_id} ---")
                print("state:", t["state"])
                print("action:", t["action"], " reward:", t["reward"])
                print("[Actor] probs:", t["actor"]["probs"][0])
                print("[Critic] V(s):", t["critic_s"]["v"][0], "  V(s'):", t["critic_s2"]["v"][0])
                print("target:", t["target"], "  TD:", t["td"])
                print("loss_actor:", t["actor_loss"], "  loss_critic:", t["critic_loss"])

            state = next_state
            ep_ret += float(reward)

        scores.append(ep_ret)
        print(f"Episode {ep:03d} | Return: {ep_ret:.1f}")

    env.close()
