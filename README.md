# Dueling DQN on LunarLander

A PyTorch implementation of a **Dueling Deep Q-Network** (Dueling DQN) that
learns to land the lunar module of Gymnasium's
[LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

The code is commented as a course: once you have read this README, read
[dualing_dqn_torch.py](dualing_dqn_torch.py) from top to bottom.

- [Project layout](#project-layout)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Concepts: from Q-learning to Dueling DQN](#concepts-from-q-learning-to-dueling-dqn)
- [Hyperparameters](#hyperparameters)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Project layout

| Path | Role |
|---|---|
| `dualing_dqn_torch.py` | Training: replay buffer, dueling network, agent, training loop. |
| `run_dualing_dqn_torch.py` | Inference: loads a trained checkpoint and plays episodes. |
| `requirements.txt` | Python dependencies. |
| `logs` | Log file, appended to by every training and inference run. |
| `tmp/dqn/<run-id>/` | One folder per training run: checkpoints and learning curve. |

## Installation

Requirements: **Python 3.10 to 3.13** (tested with 3.10) on Linux, macOS or
Windows. A GPU is optional: the network is small and trains fine on CPU.

### One command: `make install` (Linux, macOS)

```shell
make install
```

It creates `.venv`, installs the PyTorch build that matches your hardware,
then the rest of `requirements.txt`, and finally checks that PyTorch and
LunarLander work:

| Hardware detected | PyTorch build installed |
|---|---|
| NVIDIA GPU (`nvidia-smi` lists a GPU) | CUDA (`cu126`) |
| AMD GPU (`rocminfo` lists a `gfx` agent) | ROCm (`rocm6.4`) |
| No GPU, or macOS | CPU only (on macOS: the PyPI build, with Apple MPS support) |

To force a build, or change a version:

```shell
make cpu_install                   # CPU only, even if a GPU is present
make cuda_install CUDA=cu128       # another CUDA version: cu126, cu128, cu129
make rocm_install ROCM=rocm6.3     # another ROCm version: rocm6.3, rocm6.4
make install VENV=path/to/venv     # another venv location
```

`make` uses [uv](https://docs.astral.sh/uv/) if it is installed: it then
creates the venv with Python 3.10 and downloads it if needed (choose
another with `PYTHON=3.12`). Without uv it uses `python -m venv` and `pip`
with the newest `python3.13` to `python3.10` found on your `PATH`. PyTorch
2.8.0 does not support Python 3.14 or newer; `make` stops with a message
if the venv uses one.

On **Linux ARM** (Raspberry Pi, NVIDIA Jetson, ARM cloud servers), where no
prebuilt Box2D exists, `make` compiles `box2d-py` instead (see
[Linux ARM](#linux-arm-compiling-box2d)). It installs SWIG in the venv by
itself, but first checks that a C++ compiler and the Python headers are
installed, and stops with the `apt install` command to run if they are
not. `make install BOX2D_BUILD=yes` forces this build on any Linux or
macOS machine.

Other targets: `make check` (test the installation), `make train`,
`make run DIR=tmp/dqn` (inference), `make help`.

The manual installation steps are below.

### With `venv` and `pip`

```shell
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### With `uv`

```shell
uv venv -p 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Gymnasium and Box2D

LunarLander is a **Box2D** environment: besides `gymnasium` itself, it
needs the Box2D physics engine and `pygame-ce` for the window. Plain
`pip install gymnasium` does **not** install them, and `gym.make('LunarLander-v3')`
then fails with `DependencyNotInstalled: Box2D is not installed`. Install
Gymnasium with an extra instead.

**Option 1: only Box2D (what `requirements.txt` does, recommended)**

```shell
pip install "gymnasium[box2d]==1.3.0"      # or: uv pip install "gymnasium[box2d]==1.3.0"
```

The `box2d` extra installs the `box2d` package, which has prebuilt wheels
for Python 3.9 to 3.13 on Linux x86_64/i686, macOS (Intel and Apple
Silicon) and Windows: nothing is compiled and SWIG is not needed.
`box2d` has **no source package**, so it cannot be installed on any other
platform, in particular Linux ARM: see
[Linux ARM](#linux-arm-compiling-box2d).

**Option 2: every Gymnasium environment (`gymnasium[all]`)**

To also get Atari, MuJoCo and the other environment families:

1. Install SWIG and a C/C++ compiler. The `all` extra pulls in `box2d-py`,
   which has no prebuilt wheel for recent Python versions and is compiled
   from source with SWIG.

   ```shell
   sudo apt install swig build-essential python3-dev   # Debian / Ubuntu / Kali
   brew install swig                                   # macOS
   ```

   Use the system SWIG: `pip install swig` in the venv is not enough here,
   because pip compiles `box2d-py` in an isolated build environment where
   that `swig` command fails (`ModuleNotFoundError: No module named 'swig'`).

2. Then install Gymnasium:

   ```shell
   uv pip install "gymnasium[all]"            # or: pip install "gymnasium[all]"
   ```

`gymnasium[all]` is a much larger download (it also installs JAX, MuJoCo,
OpenCV, ...) and is not needed for this project.

### Linux ARM: compiling Box2D

On Linux ARM (`uname -m` prints `aarch64` or `armv7l`), `gymnasium[box2d]`
cannot be installed: its `box2d` package exists only as prebuilt wheels
for other platforms. Use `box2d-py` instead, which provides the same
`Box2D` module and is compiled from source. `make install` does all of
this automatically; by hand, in the activated venv:

```shell
# 1. System build tools (C++ compiler, Python headers):
sudo apt install build-essential python3-dev

# 2. SWIG and the build tools in the venv, then compile box2d-py with them
#    (--no-build-isolation, so that the build uses this swig):
pip install "swig==4.*" setuptools wheel
pip install --no-build-isolation "box2d-py==2.3.5"

# 3. The other requirements, with plain gymnasium instead of gymnasium[box2d]:
sed 's/^gymnasium\[box2d\]/gymnasium/' requirements.txt > requirements-arm.txt
pip install -r requirements-arm.txt "pygame-ce>=2.1.3"
```

With uv, replace `pip install` by `uv pip install`.

Note: this project uses the **Gymnasium** API (`gymnasium`, the maintained
fork of OpenAI Gym), not the old `gym` package. `env.reset()` returns
`(observation, info)` and `env.step()` returns five values
`(observation, reward, terminated, truncated, info)`; the old `gym`
returns different tuples and does not work with this code.

### CPU-only PyTorch (smaller download)

On Linux, `torch` from PyPI is the CUDA build (several GB). If you have no
NVIDIA GPU, install the CPU build **before** the requirements:

```shell
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Replace `cpu` with `cu126` (CUDA 12.6) or `rocm6.3` (AMD) to pick another
build; see [pytorch.org](https://pytorch.org/get-started/locally/).

### Check the installation

```shell
python -c "import gymnasium as gym; env = gym.make('LunarLander-v3'); print(env.reset(seed=0)[0])"
```

It prints the 8 numbers of the initial state.

## Training

```shell
python dualing_dqn_torch.py
```

The settings are constants in the `__main__` block of
`dualing_dqn_torch.py`:

| Setting | Default | Meaning |
|---|---|---|
| `num_games` | `1000` | Number of training episodes. |
| `RENDER` | `False` | `True` shows the game in a window. Rendering is capped at 50 FPS, so training becomes much slower (1 to 2 hours for 1000 episodes). |
| `load_checkpoint` | `False` | `True` starts from saved weights instead of a random network. See the note below. |
| `Agent(...)` | | Learning hyperparameters, see [Hyperparameters](#hyperparameters). |

### What a run produces

Each run gets an id (`<date>-<time>-<process id>`, e.g.
`20261001-050330-1208731`) and writes:

- `tmp/dqn/<run-id>/q_eval_dueling_dqn` and `q_next_dueling_dqn`: the
  weights of the two networks. They are saved each time the average score
  of the last 100 episodes reaches a new best, so they hold the **best**
  model of the run, not the last one.
- `tmp/dqn/<run-id>/lunar_lander_dueling.png`: the learning curve, saved at
  the end of training. It shows epsilon (blue, left axis) and the running
  average score (orange, right axis).
- Lines in `logs` (and on the terminal), one per episode:

```
2026-10-01 05:03:33 [INFO] Dualing DQN [20261001-050330-1208731]: episode: 0 - score: -40.0 - average score: -40.0 - epsilon: 1.00 - steps: 84 - end: crashed
```

| Field | Meaning |
|---|---|
| `[20261001-...]` | Run id. Lets you separate runs that wrote to `logs` at the same time: `grep <run-id> logs`. |
| `score` | Sum of the rewards of the episode. **+200 means the task is solved.** |
| `average score` | Mean score of the last 100 episodes, the standard measure of progress. |
| `epsilon` | Current probability of a random action (exploration). |
| `steps` | Length of the episode (at most 1000). |
| `end` | `landed`, `crashed`, or `timeout` (the 1000-step limit was reached). |

Note: `load_checkpoint = True` loads from the folder of the **new** run,
which is empty. To resume an earlier run, pass that run's folder as
`chkpt_dir` to `Agent(...)`, e.g. `chkpt_dir='tmp/dqn/20261001-050330-1208731'`.

### What to expect

Negative scores are normal for a long time. A random agent scores about
-150 to -200, and a crash costs -100. A typical run goes through these
stages:

1. **Exploration** (about the first 100 episodes): epsilon goes from 1.0
   to 0.01, the agent acts mostly at random, episodes end in `crashed`,
   with scores from -100 to -500.
2. **Hovering**: the agent has learned that crashing is expensive, so it
   stays in the air until the time limit. Episodes end in `timeout` after
   about 1000 steps, with scores a little below 0 (fuel costs).
3. **Landing**: the agent learns to put the module down. More and more
   episodes end in `landed` and the average score rises towards +200.

In the runs logged during development, positive scores (up to +250)
appeared after about 550 episodes. If the average is still below 0 after
1000 episodes, increase `num_games`.

## Inference

```shell
python run_dualing_dqn_torch.py path/to/dir [--episodes N] [--no-render]
```

`path/to/dir` is either:

- the folder of one run, e.g. `tmp/dqn/20261001-050330-1208731`: its
  `q_eval_dueling_dqn` is loaded;
- a folder of runs, e.g. `tmp/dqn`: the most recently saved
  `q_eval_dueling_dqn` among its subfolders is loaded.

| Option | Default | Meaning |
|---|---|---|
| `--episodes N` | `5` | Number of episodes to play. |
| `--no-render` | off | Play without opening a window. |

The agent always takes its best action (no exploration). Each episode
prints its score, step count and how it ended, and the last line gives the
average score. For a well-trained agent the output looks like this
(illustrative numbers):

```
INFO: checkpoint: tmp/dqn/20261001-050330-1208731/q_eval_dueling_dqn
INFO: episode: 0 - score: 243.1 - steps: 312 - end: landed
...
INFO: average score over 5 episodes: 221.7
```

Only `q_eval` is needed for inference: `q_next` is used only during
training.

## Concepts: from Q-learning to Dueling DQN

### 1. Reinforcement learning

An **agent** interacts with an **environment**. At each time step $t$ it
observes a state $s_t$, takes an action $a_t$, receives a reward $r_t$ and
moves to a new state $s_{t+1}$. Its goal is to maximise the **return**,
the sum of future rewards discounted by $\gamma \in [0, 1)$:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots$$

A reward now is worth more than the same reward later. With
$\gamma = 0.99$, a reward 100 steps ahead still counts for about 37 % of
its value, so the agent plans far ahead.

### 2. LunarLander

| | |
|---|---|
| State (8 numbers) | x, y position; x, y velocity; angle; angular velocity; left leg touching the ground (0/1); right leg touching the ground (0/1). |
| Actions (4) | 0 do nothing, 1 fire left engine, 2 fire main engine, 3 fire right engine. |
| Rewards | Moving towards the pad and slowing down earns reward, moving away costs reward; +10 per leg touching the ground; -0.3 per step of main engine, -0.03 per step of side engine; **+100** for landing and coming to rest, **-100** for crashing. |
| End of episode | Landing, crash or leaving the screen (`terminated`), or 1000 steps (`truncated`). |
| Solved | Score of +200 per episode. |

### 3. Q-learning

The **action-value function** $Q(s, a)$ is the expected return when the
agent takes action $a$ in state $s$ and plays well afterwards. Once $Q$ is
known, acting well is trivial: take $\arg\max_a Q(s, a)$.

$Q$ satisfies the **Bellman equation**: the value of $(s, a)$ is the
immediate reward plus the discounted value of the best action in the next
state $s'$:

$$Q(s, a) = r + \gamma \max_{a'} Q(s', a')$$

Q-learning turns this equation into a learning rule. The right-hand side,
the **TD target** $y$ (TD for temporal difference), is used as the label
that $Q(s, a)$ must move towards. When $s'$ is terminal there is no
future, and $y = r$.

### 4. DQN: Q-learning with a neural network

LunarLander's states are continuous, so $Q$ cannot be stored in a table.
**DQN** (Mnih et al., 2015) uses a neural network $Q_\theta(s, a)$ trained
to minimise the squared error between prediction and target:

$$L(\theta) = \big(y - Q_\theta(s, a)\big)^2$$

Training a network on its own predictions is unstable. DQN adds two
mechanisms:

- **Experience replay** (`ReplayBuffer`): every transition
  $(s, a, r, s', \text{done})$ is stored in a large memory, and each
  learning step uses a **random minibatch** of past transitions.
  Consecutive frames are nearly identical. Sampling at random breaks this
  correlation and lets each experience be reused many times.
- **Target network** (`Agent.q_next`): the target $y$ is computed with a
  frozen copy $Q_{\theta^-}$ of the network, refreshed every `replace`
  learning steps. Without it, every update would also move the target,
  and the network would chase its own tail:

$$y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$$

To explore, the agent follows an **epsilon-greedy** policy: with
probability $\epsilon$ it takes a random action, otherwise the best
action according to $Q$. $\epsilon$ starts at 1 (all random) and
decreases to 0.01 during training.

### 5. Dueling DQN: splitting Q into value and advantage

**Dueling DQN** (Wang et al., 2016) keeps everything from DQN and changes
only the **architecture** of the network. $Q$ is split into:

- the **state value** $V(s)$: how good it is to be in state $s$,
  whatever the action;
- the **advantage** $A(s, a)$: how much better action $a$ is than the
  average action in $s$.

The network has a shared body and two heads:

```
                    state s (8)
                        |
             fc1: Linear(8, 128) + ReLU
                        |
            fc2: Linear(128, 128) + ReLU      <- shared features
                 /                 \
     V: Linear(128, 1)      A: Linear(128, 4)
            |                       |
         V(s) (1)              A(s, .) (4)
                 \                 /
          Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')
```

**Why subtract the mean of the advantages?** $V + A$ alone is not
identifiable: adding a constant $c$ to $V$ and removing $c$ from every
$A$ gives the same $Q$, so nothing forces the network to put the value of
the state in $V$. Forcing the advantages to average to zero removes this
freedom. $V(s)$ then really is the mean Q-value of the state, and $A$ the
relative gain of each action:

$$Q(s, a) = V(s) + \Big(A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')\Big)$$

**Why does it help?** In many states the action barely matters. For
example, when the lander is high in the sky and falling slowly, every
action leads to a similar future. A plain DQN must learn the value of such
a state separately for every action. The dueling network learns it once
in $V$, and every update, whatever the action, improves $V$ for all
actions. The advantage head only has to learn the (often small)
differences between actions. Learning is faster and the value estimates
are more robust.

**Choosing an action** only needs $A$: $V(s)$ and the mean of $A$ are the
same for every action, so $\arg\max_a Q(s, a) = \arg\max_a A(s, a)$.

### 6. The algorithm, as implemented

```
initialise q_eval with random weights, q_next = copy of q_eval
epsilon = 1.0
for each episode:
    s = env.reset()
    until the episode ends:
        a = random action with probability epsilon, else argmax A(s, .)
        s', r, terminated, truncated = env.step(a)
        store (s, a, r, s', terminated) in the replay buffer
        learn():
            every `replace` steps: q_next = copy of q_eval
            sample a minibatch of 64 transitions
            Q_pred   = V(s) + A(s, a) - mean A(s, .)                 [q_eval]
            y        = r + gamma * max_a' Q_next(s', a')             [q_next]
                       (y = r if s' is terminal)
            gradient step on (y - Q_pred)^2
            epsilon = max(epsilon - eps_dec, eps_min)
        s = s'
```

Two details that matter:

- Only `terminated` is stored as "done". When an episode is cut by the
  1000-step limit (`truncated`), the lander still had a future, so its
  target keeps the value of $s'$.
- On a terminal transition only the future term is zeroed. The last
  reward (+100 for landing, -100 for a crash) stays in the target, because
  it is the most informative reward of the episode.

## Hyperparameters

Set in `Agent(...)` in the `__main__` block of `dualing_dqn_torch.py`.

| Name | Value | Role |
|---|---|---|
| `gamma` | 0.99 | Discount factor. Close to 1 because landing can be hundreds of steps away. |
| `alpha` | 5e-4 | Learning rate of Adam. |
| `epsilon` | 1.0 | Initial exploration rate. |
| `eps_min` | 0.01 | Final exploration rate. |
| `eps_dec` | 5e-5 | Decrease of epsilon per learning step. Exploration lasts about 20 000 steps (roughly the first 100 episodes). Too large a value stops exploration before the agent has ever landed. |
| `batch_size` | 64 | Transitions per learning step. |
| `mem_size` | 1 000 000 | Replay buffer capacity. |
| `replace` | 100 | Learning steps between two copies of `q_eval` into `q_next`. |

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `AttributeError: module 'numpy' has no attribute 'bool'` | numpy 1.24 to 1.26 is installed. Install numpy 2: `pip install "numpy>=2"`. |
| `DependencyNotInstalled: Box2D is not installed` | Gymnasium was installed without the Box2D extra. See [Gymnasium and Box2D](#gymnasium-and-box2d). |
| `box2d==2.3.10 has no source distribution` (uv) or `No matching distribution found for box2d==2.3.10` (pip) | No prebuilt Box2D for your platform (e.g. Linux ARM). Use `make install`, or follow [Linux ARM](#linux-arm-compiling-box2d). |
| `error: command 'swig' failed` (or `swig: not found`) while installing | `box2d-py` (from `gymnasium[all]`) is being compiled and SWIG is missing. Install it (`sudo apt install swig build-essential python3-dev`), or use `gymnasium[box2d]`, which needs no compilation. |
| `ModuleNotFoundError: No module named 'swig'` while building `box2d-py` | SWIG was installed with pip, but pip builds in an isolated environment where it cannot run. Use the system SWIG, or build with `--no-build-isolation` as in [Linux ARM](#linux-arm-compiling-box2d). |
| No window, or an error about the display, with `RENDER = True` | No graphical display (SSH, Docker). Train with `RENDER = False`, and run inference with `--no-render`. |
| Training is very slow | `RENDER = True` limits training to 50 steps per second. Set it to `False`. |
| Scores stay negative | Normal for the first hundreds of episodes, see [What to expect](#what-to-expect). Check the `end` field in `logs`: many `timeout` endings mean the agent hovers and needs more episodes. |

## References

- V. Mnih et al., *Human-level control through deep reinforcement
  learning*, Nature, 2015 (DQN).
- Z. Wang et al., *Dueling Network Architectures for Deep Reinforcement
  Learning*, ICML 2016, [arXiv:1511.06581](https://arxiv.org/abs/1511.06581).
- R. Sutton and A. Barto, *Reinforcement Learning: An Introduction*, 2nd
  edition, 2018, [free online](http://incompleteideas.net/book/the-book-2nd.html).
- [Gymnasium LunarLander documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/).
