# 🍄 Mario AI Bot — Self-Playing Mario with Reinforcement Learning

An AI that learns to play Mario Bros completely by itself using Deep Reinforcement Learning (PPO algorithm). Watch it go from a complete beginner to beating levels on its own!

![Mario AI](https://upload.wikimedia.org/wikipedia/en/a/a9/MarioNSMBUDeluxe.png)

## 🎮 Demo
The AI starts knowing nothing and learns purely through trial and error — just like a human would!

- ✅ Learns to move right
- ✅ Learns to jump over obstacles  
- ✅ Gets better every hour of training
- ✅ Can watch it learn in real time

## 🧠 How It Works
This project uses:
- **PPO (Proximal Policy Optimization)** — a powerful reinforcement learning algorithm
- **CNN (Convolutional Neural Network)** — lets the AI "see" the game screen
- **OpenAI Gymnasium + ALE** — the game environment
- **Stable Baselines3** — the AI training framework

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOURUSERNAME/mario-ai-bot.git
cd mario-ai-bot
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
AutoROM --accept-license
```

### 4. Run the bot
```bash
python mario_bot.py
```

### 5. Choose an option
```
1. Train AI - hidden window (FASTER)
2. Train AI - visible window (watch Mario learn)
3. Watch trained AI play Mario
```

## ⏱️ Training Time
| Timesteps | Time | Skill Level |
|-----------|------|-------------|
| 100,000 | ~15 mins | Barely moves |
| 500,000 | ~1 hour | Learning to jump |
| 1,000,000 | ~2-3 hours | Getting decent |
| 5,000,000 | ~10 hours | Pretty good! |

## 💻 Requirements
- Python 3.11
- Windows 10/11
- 8GB RAM minimum
- No GPU needed!

## 📦 Tech Stack
- Python 3.11
- PyTorch 2.0.0
- Stable Baselines3
- Gymnasium + ALE
- OpenCV

## 🌟 Give it a Star!
If you found this cool or useful, please give it a ⭐ — it helps others find the project!

## 📜 License
MIT License — free to use for anything!