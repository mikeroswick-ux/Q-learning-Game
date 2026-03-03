# Q-learning-Game
An interactive game where a **Deep Q-Network (DQN)** agent learns to fight a boss using reinforcement learning, entirely in the browser with TensorFlow.js.


## Features

-  **Real-time game** – Player (controlled by DQN) vs. AI boss.
-  **DQN agent** – Implements experience replay, target network, and epsilon‑greedy exploration.
-  **Live training** – Watch the agent improve over episodes.
-  **Speed control** – Run the game at 0.5x to 5x speed.
-  **Model persistence** – Save/load trained models to/from browser’s LocalStorage.
-  **Reward breakdown & plot** – Detailed scoreboard and a live reward curve.
-  **Adjustable parameters** – Change epsilon, replay start, etc. directly in the code.

## Demo

<img width="2889" height="1522" alt="image" src="https://github.com/user-attachments/assets/ac603cc7-9add-434c-b06a-007a399d28ab" />


## How It Works

The player agent observes a **9‑dimensional state vector** (relative positions, distances, boss state, etc.) and chooses one of **10 discrete actions** (move, attack, dodge). The boss follows a hand‑coded heuristic. The agent receives shaped rewards for staying alive, hitting the boss, dodging, etc., and a large terminal reward when winning.

The DQN is built with TensorFlow.js, uses two hidden layers (64 units each), and is trained with mini‑batch gradient descent. Experience is stored in a replay buffer.

## Getting Started

### Prerequisites

- A modern web browser (Chrome, Firefox, Edge).
- (Optional) A local web server for best performance – because of `module` loading and CORS, you should serve the files.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dqn-boss-battle.git
   cd dqn-boss-battle
   ```

2. Start a simple HTTP server (e.g., Python):
   ```bash
   python -m http.server 8000
   ```

3. Open `http://localhost:8000` in your browser.

> **Note**: The game relies on TensorFlow.js loaded from a CDN, so an internet connection is required.

## Usage

### Controls

| Button          | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| **▶ Start**     | Start/pause the game loop.                                                  |
| **⚙ Train**     | Toggle training mode. When on, the game runs continuously and the agent learns from each episode. |
| **🔄 Reset**     | Reset everything: agent, game state, episode counter, and scoreboard.      |
| **Speed buttons** | Change the simulation speed (0.5x – 5x).                                  |
| **📈 Show Plot** | Draw the reward curve for the last N episodes (N set in the input field).  |
| **💾 Save Model**| Save the current DQN model to browser LocalStorage.                        |
| **📂 Load Model**| Load a previously saved model from LocalStorage.                           |

The **scoreboard** on the right shows a detailed breakdown of rewards earned during the current episode. The **status bar** below the canvas displays episode number, memory size, current epsilon, and remaining battle time.

### Training

- Press **⚙ Train** to start training. The agent will run episodes continuously, updating its policy.
- You can pause at any time with the **Start/Pause** button.
- Watch the epsilon value decay from 1.0 to 0.05 as training progresses.
- The plot updates every time the episode count reaches the value in the “显示回合数” (plot after) field.

### Saving and Loading

- Use **💾 Save Model** to store the current model weights in your browser’s LocalStorage.
- Use **📂 Load Model** to restore a previously saved model. The target network is also updated automatically.

## Project Structure

```
├── index.html          # Main HTML page
├── style.css           # All styling
├── js/
│   ├── utils.js        # Utility functions (clamp, dist, angleTo, colors)
│   ├── game.js         # Game entities (Entity class) and drawing helpers
│   ├── dqn.js          # DQNAgent class with TensorFlow.js model
│   └── main.js         # Game loop, state handling, reward logic, UI events
```

## DQN Agent Details

- **State space**: 9 continuous features (normalized coordinates, distances, boss attack state, etc.)
- **Action space**: 10 discrete actions – move in four directions, stay, attack, dodge in four directions.
- **Neural network**: 2 hidden layers of 64 neurons with ReLU, output layer linear.
- **Hyperparameters** (in `main.js`):
  - Learning rate: 0.001
  - Discount factor γ: 0.95
  - Initial ε: 1.0, final ε: 0.05, decay: 0.995 per episode
  - Batch size: 64
  - Replay memory size: 20,000
  - Train start: 1,000 samples
  - Target network sync interval: 500 steps

## Reward Design

The reward is shaped to encourage good behavior:

| Component        | Typical value | Description                                   |
|------------------|---------------|-----------------------------------------------|
| Survival         | +0.02 per step | Small positive for staying alive.             |
| Proximity        | ±0.5          | Bonus when at optimal distance (100–200px), penalty too close or far. |
| Boundary penalty | –             | Negative when near edges.                     |
| Center bonus     | +0.2          | Extra when player stays in the central region.|
| Attack range     | +0.5          | Bonus for being in attack range but not attacking. |
| Hit boss         | +40           | Successfully hitting the boss.                |
| Hit by boss      | –50           | Getting hit by the boss.                      |
| Miss             | –5            | Attacking and missing.                        |
| Dodge            | +3            | Boss attacks while player is outside the sector. |
| Terminal         | ±1000 / ±20   | Large positive for winning, small negative for losing. |

## Customization

You can tweak the game or the agent by modifying the constants at the top of `main.js`:

- `ACTIONS` – change available actions.
- `HYPERPARAMS` – adjust learning rate, epsilon decay, etc.
- `BOSS_WINDUP`, `PLAYER_WINDUP` – change attack timings.
- Reward values in `processRewardAndTrain()`.

---

**Have fun training your DQN boss slayer!** 🤖⚔️
