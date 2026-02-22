# ============================================================
# MARIO AI BOT - Fixed ALE Version
# ============================================================

import ale_py
import gymnasium as gym
import gymnasium.envs.registration
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
import numpy as np
import cv2
import os

# This line registers all Atari games including Mario
gym.register_envs(ale_py)

# ============================================================
# LIVE WINDOW - So you can watch the game!
# ============================================================

class LiveRenderWrapper(gym.Wrapper):
    """
    Shows the game window while training.
    Press Q on the window to close it anytime.
    Training will continue even if you close the window.
    """
    def __init__(self, env, window_name="Mario AI"):
        super().__init__(env)
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 600, 500)

    def step(self, action):
        result = self.env.step(action)
        self._show_frame()
        return result

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._show_frame()
        return result

    def _show_frame(self):
        frame = self.env.render()
        if frame is not None and len(frame.shape) == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.window_name, frame_bgr)
            key = cv2.waitKey(1)
            if key == ord('q'):
                print("👋 Window closed. Training continues!")
                cv2.destroyWindow(self.window_name)

    def close(self):
        cv2.destroyAllWindows()
        self.env.close()


# ============================================================
# BUILD ENVIRONMENT
# ============================================================

def make_env(show_window=False):
    """
    Creates the Mario game environment.
    show_window=True means you can see the game playing!
    """
    if show_window:
        env = gym.make("ALE/MarioBros-v5", render_mode="rgb_array")
        env = LiveRenderWrapper(env)
    else:
        env = gym.make("ALE/MarioBros-v5", render_mode=None)

    # Handles grayscale, resize, frame skip automatically
    env = AtariWrapper(env)
    return env


def create_vec_env(show_window=False):
    print("🎮 Creating Mario environment...")
    env = DummyVecEnv([lambda: make_env(show_window)])
    env = VecFrameStack(env, n_stack=4)
    print("✅ Environment ready!")
    return env


# ============================================================
# TRAINING CALLBACK - Shows progress and saves automatically
# ============================================================

class TrainingCallback(BaseCallback):
    def __init__(self, check_freq=1000, save_path='./models/'):
        super().__init__(verbose=0)
        self.check_freq = check_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            print(f"✅ Steps completed: {self.n_calls} / {self.model._total_timesteps}")
            path = os.path.join(self.save_path, f'mario_checkpoint_{self.n_calls}')
            self.model.save(path)
            print(f"💾 Progress saved!")
        return True


# ============================================================
# TRAIN THE AI
# ============================================================

def train_mario(show_window=False):
    print("\n" + "="*50)
    print("🚀 STARTING MARIO AI TRAINING")
    print("="*50)

    env = create_vec_env(show_window=show_window)

    if os.path.exists('./models/mario_final.zip'):
        print("📂 Found saved model! Continuing from last session...")
        model = PPO.load('./models/mario_final', env=env)
        print("✅ Loaded successfully!")
    else:
        print("🆕 No saved model found. Starting fresh...")
        model = PPO(
            policy='CnnPolicy',
            env=env,
            verbose=1,
            learning_rate=0.000001,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log='./logs/'
        )

    callback = TrainingCallback(check_freq=1000, save_path='./models/')

    print("\n⏳ Training started!")
    if show_window:
        print("👀 Watch the game window to see Mario learning!")
        print("💡 At first Mario will be terrible - that is totally normal!")
        print("💡 Press Q on the game window anytime to hide it")
    print("💡 Press CTRL+C anytime to stop - progress is saved!\n")

    try:
        model.learn(
            total_timesteps=1_000_000,
            callback=callback,
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("\n⛔ Training stopped by you.")

    model.save('./models/mario_final')
    print("💾 Final model saved as mario_final!")
    env.close()


# ============================================================
# WATCH THE AI PLAY
# ============================================================

def watch_mario_play(num_episodes=5):
    print("\n" + "="*50)
    print("👀 WATCHING MARIO AI PLAY")
    print("="*50)

    if not os.path.exists('./models/mario_final.zip'):
        print("❌ No trained model found!")
        print("Please train first by choosing option 1 or 2.")
        return

    env = create_vec_env(show_window=True)
    model = PPO.load('./models/mario_final', env=env)
    print("✅ Model loaded! Watch the game window!")

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        print(f"\n🎮 Playing episode {episode + 1} of {num_episodes}...")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1

        print(f"✅ Episode {episode + 1} done!")
        print(f"   Reward: {total_reward:.1f} | Steps: {steps}")

    env.close()
    print("\n🎮 Finished watching!")


# ============================================================
# MAIN MENU
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*40)
    print("       🍄 MARIO AI BOT 🍄")
    print("="*40)
    print("1. Train AI - hidden window (FASTER)")
    print("2. Train AI - visible window (watch Mario learn)")
    print("3. Watch trained AI play Mario")
    print("="*40)

    choice = input("\nEnter 1, 2, or 3: ").strip()

    if choice == "1":
        train_mario(show_window=False)
        again = input("\nWatch AI play now? (yes/no): ").strip().lower()
        if again == "yes":
            watch_mario_play()

    elif choice == "2":
        train_mario(show_window=True)
        again = input("\nWatch AI play now? (yes/no): ").strip().lower()
        if again == "yes":
            watch_mario_play()

    elif choice == "3":
        watch_mario_play()

    else:
        print("❌ Please enter 1, 2, or 3 only.")
