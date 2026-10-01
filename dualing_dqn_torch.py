"""Dueling Deep Q-Network (Dueling DQN) trained on LunarLander-v3.

This file is meant to be read from top to bottom as a small course. Each
building block of the method has its own class:

  ReplayBuffer  -> stores past experience          (from DQN)
  Model         -> the dueling neural network      (the "dueling" part)
  Agent         -> acts, learns, keeps a target network (from DQN)
  __main__      -> the training loop

1. The problem: reinforcement learning
--------------------------------------
An agent plays in an environment. At each time step t it:

  - observes a state s_t          (here: 8 numbers describing the lander)
  - chooses an action a_t         (here: 4 choices, see below)
  - receives a reward r_t         (a number: good or bad for the agent)
  - lands in a new state s_{t+1}

The goal is to pick actions that maximise the *return*, the sum of future
rewards discounted by a factor gamma (0 < gamma < 1):

  G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

gamma < 1 means "a reward now is worth more than the same reward later",
and it keeps the sum finite. With gamma = 0.99, a reward 100 steps ahead
still counts for 0.99^100 ~ 37% of its value.

2. Q-learning
-------------
The action-value function Q(s, a) is the expected return when the agent
takes action a in state s and then plays well. If Q is known, acting well
is trivial: in state s, take argmax_a Q(s, a).

Q satisfies the Bellman equation: the value of (s, a) is the immediate
reward plus the discounted value of the best action in the next state:

  Q(s, a) = r + gamma * max_a' Q(s', a')

Q-learning turns this equation into a learning rule: the right-hand side,
called the TD target (TD = temporal difference), is used as the "label"
that the left-hand side must move towards.

3. DQN (Deep Q-Network, Mnih et al., 2015)
------------------------------------------
The states of LunarLander are continuous, so Q cannot be stored in a
table: a neural network approximates it. Training a network on its own
predictions is unstable, so DQN adds two tricks:

  - Experience replay (see `ReplayBuffer`): transitions (s, a, r, s', done)
    are stored and the network learns from random minibatches of them.
    This breaks the correlation between consecutive steps and lets each
    experience be reused many times.
  - Target network (see `Agent.q_next`): the TD target is computed with a
    frozen copy of the network, refreshed only every `replace` learning
    steps. Otherwise the target would move at every update, like chasing
    one's own tail.

4. Dueling DQN (Wang et al., 2016)
----------------------------------
The dueling idea changes only the *architecture* of the network. Q is
split into two parts:

  V(s)     the state value: how good it is to be in state s, whatever
           the action.
  A(s, a)  the advantage: how much better action a is than the average
           action in state s.

The network has a shared body and two heads, one for V (1 output) and one
for A (one output per action), recombined into Q:

  Q(s, a) = V(s) + ( A(s, a) - mean_a' A(s, a') )

Why subtract the mean? V + A alone is not "identifiable": adding a
constant c to V and removing c from every A gives exactly the same Q, so
the network has no reason to put the state value in V. Forcing the
advantages to average to zero removes that freedom: V(s) then really is
the average Q-value of the state, and A really is the relative gain of
each action.

Why does it help? In many states the action barely matters (the lander
is high in the sky and falling slowly: every action leads to a similar
future). A plain DQN must learn the value of such a state separately for
each action; the dueling network learns it once in V, and every update of
any action improves V for all actions. The advantage head only has to
learn the (often small) differences between actions.

Everything else (replay buffer, target network, epsilon-greedy
exploration, TD target) is the same as in DQN.

5. The environment: LunarLander-v3 (gymnasium)
----------------------------------------------
  State (8 numbers): x, y position; x, y velocity; angle; angular
    velocity; left leg touches ground (0/1); right leg touches ground (0/1).
  Actions (4): 0 do nothing, 1 fire left engine, 2 fire main engine,
    3 fire right engine.
  Rewards: moving closer to the landing pad and slowing down earns reward,
    moving away costs reward; each leg contact +10; firing the main
    engine -0.3 per step, a side engine -0.03 per step; landing and coming
    to rest +100; crashing -100.
  An episode ends when the lander lands, crashes or flies off screen
  (`terminated`), or after 1000 steps (`truncated`).
  A score of +200 per episode counts as solving the task.
"""
import os
import sys
import time
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from loguru import logger

# Logging: messages go to the terminal and are appended to `./logs`.
# Every line of `./logs` carries the id of the run that wrote it, so the
# episodes of runs launched at the same time can be told apart in the file.
RUN_ID = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
LOG_FORMAT = (
  "{time:YYYY-MM-DD HH:mm:ss} [{level}] {extra[name]} [{extra[run]}]: {message}")
logger.remove()
logger.configure(extra={"name": "Dualing DQN", "run": RUN_ID})
logger.add(sys.stdout, level="INFO", format="{level}: {message}")
logger.add('./dualing_dqn_torch.log', level="INFO", format=LOG_FORMAT)


class ReplayBuffer:
  """Experience replay memory: the agent's record of past transitions.

  A transition is one step of experience: (s, a, r, s', done), "in state s
  I took action a, received reward r, landed in state s', and the episode
  did / did not end there".

  Why not learn from each transition right away and throw it away?

    - Consecutive steps are almost identical (the lander moves a little
      between two frames). Gradient descent assumes roughly independent
      samples; a stream of near-duplicates makes the network overfit the
      current situation and forget the rest. Sampling at random from a
      large memory mixes old and new, easy and hard situations.
    - Each environment step is expensive; storing it lets the network
      learn from it many times.

  Implementation: five pre-allocated numpy arrays (one per element of the
  transition) used as a circular buffer. When the buffer is full, the
  newest transition overwrites the oldest one.
  """

  def __init__(self, max_size: int,input_shape: Tuple[int, int]) -> None:
    """Allocate the memory.

    Args:
      max_size: how many transitions are kept at most.
      input_shape: shape of one state, here (8,).
      n_actions: number of actions. Not needed here, since actions are
        stored as integers (0..3), not as one-hot vectors.
    """
    self.mem_size = max_size
    # Total number of transitions ever stored (it keeps growing after the
    # buffer is full; `mem_cntr % mem_size` gives the slot to write to).
    self.mem_cntr = 0
    # s: one row per transition, e.g. shape (max_size, 8).
    self.state_memory = np.zeros(
      shape=(self.mem_size, *input_shape),
      dtype=np.float32
    )
    # s': the state reached after the action.
    self.new_state_memory = np.zeros(
      shape=(self.mem_size, *input_shape),
      dtype=np.float32
    )
    # a: int64 because PyTorch's `gather` (used in `learn`) needs int64
    # indices.
    self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
    # r
    self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
    # done: True when s' is a terminal state (landed or crashed).
    self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

  def store_transition(self, state, action, reward, state_, done):
    """Write one transition in the next free (or oldest) slot."""
    index = self.mem_cntr % self.mem_size
    self.state_memory[index] = state
    self.action_memory[index] = action
    self.reward_memory[index] = reward
    self.new_state_memory[index] = state_
    self.terminal_memory[index] = done

    self.mem_cntr += 1

  def sample_buffer(self, batch_size):
    """Return `batch_size` transitions picked uniformly at random.

    Only the slots filled so far are eligible (`max_mem`), and
    `replace=False` prevents the same transition from appearing twice in
    one minibatch. Each returned array has `batch_size` rows; the i-th
    rows of the five arrays form the i-th transition.
    """
    max_mem = min(self.mem_cntr, self.mem_size)
    batch = np.random.choice(max_mem, batch_size, replace=False)

    states = self.state_memory[batch]
    actions = self.action_memory[batch]
    rewards = self.reward_memory[batch]
    states_ = self.new_state_memory[batch]
    dones = self.terminal_memory[batch]

    return states, actions, rewards, states_, dones


class Model(nn.Module):
  """The dueling network: a shared body and two heads, V and A.

  Architecture (for LunarLander: 8 inputs, 4 actions):

                        state s (8)
                            |
                 fc1: Linear(8, 128) + ReLU
                            |
                fc2: Linear(128, 128) + ReLU      <- shared features
                     /                 \\
         V: Linear(128, 1)      A: Linear(128, 4)
                |                       |
             V(s) (1)             A(s, .) (4)

  The two heads see the same features: the body learns a description of
  the state that is useful both to judge the state (V) and to compare
  actions (A).

  `forward` returns V and A separately; they are combined into Q with
  Q = V + (A - mean(A)) in `Agent.learn`. The agent owns two instances of
  this class: the online network `q_eval` (trained) and the target
  network `q_next` (a frozen copy, see `Agent`).
  """

  def __init__(
      self,
      alpha: float,
      n_actions: int,
      name: str,
      input_dims: int,
      chkpt_dir: str='tmp/dqn',
  ):
    """Build the layers, the optimizer and the loss.

    Args:
      alpha: learning rate of the Adam optimizer.
      n_actions: number of actions, i.e. outputs of the advantage head.
      name: used in the checkpoint file name ('q_eval' or 'q_next').
      input_dims: size of a state vector (8 for LunarLander).
      chkpt_dir: folder where the weights are saved.
    """
    super().__init__()

    # Shared body: turns the raw state into 128 learned features.
    self.fc1 = nn.Linear(input_dims, 128)
    self.fc2 = nn.Linear(128, 128)
    # Value head: a single number per state, V(s).
    self.V = nn.Linear(128, 1)
    # Advantage head: one number per action, A(s, a).
    self.A = nn.Linear(128, n_actions)

    # Adam adapts the step size of each weight; it is the usual choice for
    # DQN-style training.
    self.optimizer = optim.Adam(self.parameters(), lr=alpha)
    # Mean squared error between predicted Q and TD target, i.e. the
    # squared Bellman error averaged over the minibatch.
    self.loss = nn.MSELoss()
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)
    self.checkpoint_dir = chkpt_dir
    self.checkpoint_file = os.path.join(
      self.checkpoint_dir, name+"_dueling_dqn")

  def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute V(s) and A(s, .) for a batch of states.

    Args:
      state: tensor of shape (batch, 8).

    Returns:
      V of shape (batch, 1) and A of shape (batch, n_actions).
      There is no activation on the heads: values and advantages can be
      any real number, positive or negative.
    """
    l1 = F.relu(self.fc1(state))
    l2 = F.relu(self.fc2(l1))
    V = self.V(l2)
    A = self.A(l2)
    return V, A

  def save_checkpoint(self):
    """Save the weights (the `state_dict`) to `checkpoint_file`."""
    logger.info("... saving checkpoint ...")
    torch.save(self.state_dict(), self.checkpoint_file)

  def load_checkpoint(self):
    """Load the weights saved by `save_checkpoint`.

    `map_location` lets a checkpoint saved on a GPU be loaded on a
    CPU-only machine (and vice versa).
    """
    logger.info("... loading checkpoint ...")
    self.load_state_dict(
      torch.load(self.checkpoint_file, map_location=self.device))


class Agent:
  """The learning agent: chooses actions and improves its network.

  It holds:
    - `memory`: the replay buffer of past transitions.
    - `q_eval`: the online network. It chooses the actions and is the
      only network trained by gradient descent.
    - `q_next`: the target network. It computes the value of the next
      state in the TD target. It is not trained; every `replace` learning
      steps it receives a copy of `q_eval`'s weights. Between two copies
      the targets stay fixed, which makes the regression problem stable.
    - `epsilon`: the exploration rate (see `choose_action`).
  """

  def __init__(
      self,
      gamma: float,
      epsilon: float,
      alpha: float,
      n_actions: int,
      input_dims: int,
      mem_size: int,
      batch_size: int,
      eps_min=0.01,
      eps_dec=5e-7,
      replace=1000,
      chkpt_dir: str='tmp/dqn',
  ) -> None:
    """Create the replay buffer and the two networks.

    Args:
      gamma: discount factor of future rewards (0.99: plan far ahead).
      epsilon: initial exploration rate (1.0: start fully random).
      alpha: learning rate of the online network.
      n_actions: number of actions (4).
      input_dims: shape of a state, as a list: [8].
      mem_size: capacity of the replay buffer.
      batch_size: number of transitions per learning step.
      eps_min: epsilon never goes below this value, so the agent always
        keeps a little exploration.
      eps_dec: how much epsilon decreases after each learning step.
      replace: copy the online weights into the target network every
        `replace` learning steps (None: never).
      chkpt_dir: folder of the checkpoint files.
    """
    self.gamma = gamma
    self.epsilon = epsilon
    self.eps_min = eps_min
    self.eps_dec = eps_dec
    self.batch_size = batch_size
    self.replace_target_cnt = replace
    self.action_space = [i for i in range(n_actions)]
    # Number of gradient steps done so far; drives the target network
    # refresh.
    self.learn_step_counter = 0

    self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
    self.q_eval = Model(
      alpha=alpha,
      input_dims=input_dims[0],
      n_actions=n_actions,
      name='q_eval',
      chkpt_dir=chkpt_dir,
    )
    # Same architecture as `q_eval`. Its own optimizer is never used.
    self.q_next = Model(
      alpha=alpha,
      input_dims=input_dims[0],
      n_actions=n_actions,
      name='q_next',
      chkpt_dir=chkpt_dir,
    )

  def store_transition(self, state, action, reward, state_, done):
    """Save one step of experience in the replay buffer."""
    self.memory.store_transition(state, action, reward, state_, done)

  def choose_action(self, observation: np.ndarray) -> int:
    """Pick an action with an epsilon-greedy policy.

    This is the exploration / exploitation trade-off:

      - with probability epsilon, take a random action (explore: try
        things the agent does not yet know are good);
      - otherwise take the action with the best estimated Q (exploit
        what has been learned).

    epsilon starts at 1.0 (all random: the untrained network knows
    nothing) and decreases during training (see `decrement_epsilon`).
    Without exploration the agent would repeat its first, arbitrary
    preferences forever and never discover how to land.

    The greedy action is the argmax of the advantage A(s, a): since
    Q(s, a) = V(s) + A(s, a) - mean(A), V(s) and mean(A) are the same
    for every action and do not change the argmax.

    The forward pass runs under `torch.no_grad()` because acting never
    needs gradients; without it PyTorch builds a useless autograd graph
    at every environment step.
    """
    if np.random.random() > self.epsilon:
      # Exploit. The network expects a batch: (8,) -> (1, 8).
      observation = observation[np.newaxis, :]
      state = torch.tensor(observation).to(self.q_eval.device)
      with torch.no_grad():
        _, advantage = self.q_eval.forward(state)
      action = torch.argmax(advantage).item()
    else:
      # Explore.
      action = np.random.choice(self.action_space)
    return action

  def replace_target_network(self):
    """Copy the online weights into the target network every
    `replace_target_cnt` learning steps (including step 0, so both
    networks start identical)."""
    if (self.replace_target_cnt is not None
        and self.learn_step_counter % self.replace_target_cnt == 0):
      self.q_next.load_state_dict(self.q_eval.state_dict())

  def decrement_epsilon(self) -> None:
    """Linear epsilon decay: from 1.0 down to `eps_min`, by `eps_dec`
    per learning step. The agent explores a lot at the start and relies
    more and more on what it has learned."""
    if self.epsilon > self.eps_min:
      self.epsilon = self.epsilon - self.eps_dec
    else:
      self.epsilon = self.eps_min

  def learn(self):
    """Do one gradient step on a random minibatch of the replay buffer.

    This is the heart of the method. For each transition (s, a, r, s',
    done) of the minibatch:

      1. Prediction, with the online network:
           Q_eval(s, a) = V(s) + A(s, a) - mean_a' A(s, a')
         keeping only the Q-value of the action actually taken, a.

      2. TD target, with the target network:

           y = r + gamma * max_a' Q_next(s', a')   if s' is not terminal
           y = r                                   if s' is terminal

         On a terminal transition there is no future, so only the
         bootstrap term (the value of s') is zeroed; the reward r is
         always kept.

      3. Loss = mean over the minibatch of (y - Q_eval(s, a))^2, and one
         Adam step on the online network.

    The target y is treated as a constant label (`detach`): gradients
    flow only through the prediction. Otherwise the update would also
    move the target, and learning would chase a moving goal.

    Learning starts as soon as the buffer holds one minibatch.
    """
    if self.memory.mem_cntr < self.batch_size:
      return
    # Reset the gradients accumulated by the previous `backward()`.
    self.q_eval.optimizer.zero_grad()
    self.replace_target_network()
    data = self.memory.sample_buffer(self.batch_size)

    # numpy -> torch, on the network's device. Shapes (B = batch_size):
    # states, new_states (B, 8); actions, rewards, dones (B,).
    states, actions, rewards, new_states, dones = data
    states = torch.tensor(states).to(self.q_eval.device)
    new_states = torch.tensor(new_states).to(self.q_eval.device)
    actions = torch.tensor(actions).to(self.q_eval.device)
    rewards = torch.tensor(rewards).to(self.q_eval.device)
    dones = torch.tensor(dones).to(self.q_eval.device)

    # V_s: (B, 1), A_s: (B, 4) for the states s, from the online network.
    V_s, A_s = self.q_eval.forward(states)
    # Same for the next states s', from the target network.
    V_s_, A_s_ = self.q_next.forward(new_states)

    # Dueling aggregation: Q = V + (A - mean(A)). `keepdim=True` keeps the
    # mean as (B, 1) so it broadcasts over the 4 actions, and V (B, 1)
    # broadcasts the same way. Result: (B, 4), one Q-value per action.
    q_pred = torch.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))
    # Keep only Q(s, a) for the action taken in each transition:
    # actions (B,) -> (B, 1) indices, gather -> (B, 1), squeeze -> (B,).
    q_pred = q_pred.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    # Q-values of all actions in the next states, (B, 4).
    q_next = torch.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

    # max_a' Q_next(s', a'): the value of the next state if the agent then
    # plays greedily, (B,). `detach` makes it a constant label.
    q_next = torch.max(q_next, dim=1)[0].detach()
    # A terminal state has no future: its value is 0. Only this term is
    # zeroed, not the whole target, so the last reward (+100 for landing,
    # -100 for a crash) is still learned; it is the most informative
    # reward of the episode.
    q_next[dones] = 0.0
    # Bellman equation: y = r + gamma * value of s'. Shape (B,).
    q_target = rewards + self.gamma * q_next

    # Squared Bellman error, then backpropagation through q_pred into the
    # two heads and the shared body of the online network.
    loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
    loss.backward()
    self.q_eval.optimizer.step()
    self.learn_step_counter += 1

    self.decrement_epsilon()

  def save_models(self):
    """Save both networks' weights."""
    self.q_eval.save_checkpoint()
    self.q_next.save_checkpoint()

  def load_models(self):
    """Load both networks' weights."""
    self.q_eval.load_checkpoint()
    self.q_next.load_checkpoint()


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    """Save a plot of epsilon and of the score along training.

    Two y-axes share the episode axis: epsilon on the left (blue line),
    and on the right (orange dots) the running average of the last 21
    episode scores, which smooths the very noisy per-episode score.
    A healthy run shows epsilon going down while the average score goes
    up towards +200.
    """
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    # `x` holds one point per episode, not per training step.
    ax.set_xlabel("Episodes", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
      running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


if __name__ == '__main__':
  # Training loop. For each episode: reset the environment, then repeat
  # act -> observe -> store -> learn until the episode ends. Learning
  # happens at every environment step, on a random minibatch of the
  # whole memory, not only on the current episode.
  import gymnasium as gym

  MAX_ITER = int(1e12)
  # Show every training episode in real time in a window. Rendering is
  # capped at the game's 50 FPS, which limits training to ~50 steps/s
  # (about 1-2 hours for 1000 episodes). It is off by default so training
  # runs at full speed; set it to True to watch the agent.
  RENDER = False

  # Each run saves its checkpoints and plot in its own folder, so two runs
  # launched at the same time do not overwrite each other's models.
  RUN_DIR = os.path.join('tmp', 'dqn', RUN_ID)
  os.makedirs(RUN_DIR, exist_ok=True)

  # With `render_mode='human'` the environment draws every step itself in
  # a single window that stays open for the whole training: after a game
  # over, `env.reset()` starts the next episode in the same window.
  env = gym.make('LunarLander-v3', render_mode='human' if RENDER else None)
  num_games = 1000
  load_checkpoint = False

  agent = Agent(
    # Discount factor: landing can be hundreds of steps away, so the
    # agent must value distant rewards.
    gamma=0.99,
    # Start with 100% random actions.
    epsilon=1.0,
    # Learning rate of Adam.
    alpha=5e-4,
    # LunarLander: 8 state numbers, 4 actions.
    input_dims=[8],
    n_actions=4,
    # Large enough to keep every transition of the whole training.
    mem_size=1_000_000,
    eps_min=0.01,
    batch_size=64,
    # Epsilon is decremented once per learning step (~100-1000 steps per
    # episode). With 1e-3 it would reach eps_min after only 1000 steps,
    # i.e. ~10 episodes, so the agent would stop exploring before it had
    # seen a single landing. With 5e-5 exploration lasts ~20k steps,
    # roughly the first 100-150 episodes.
    eps_dec=5e-5,
    # Refresh the target network every 100 learning steps.
    replace=100,
    chkpt_dir=RUN_DIR,
  )
  logger.info(
    "run started - num_games: %d - gamma: %.2f - alpha: %g - "
    "batch_size: %d - eps_dec: %g - replace: %d - render: %s - dir: %s"
    % (num_games, agent.gamma, agent.q_eval.optimizer.defaults['lr'],
       agent.batch_size, agent.eps_dec, agent.replace_target_cnt, RENDER,
       RUN_DIR))

  if load_checkpoint:
    agent.load_models()
  filename = os.path.join(RUN_DIR, 'lunar_lander_dueling.png')
  scores = []
  eps_history = []
  # Best 100-episode average seen so far; the models are saved each time
  # it improves, so the checkpoint is the best model of the run.
  best_score = -np.inf

  for i in range(num_games):
    done = False
    # The score of an episode is the plain (undiscounted) sum of its
    # rewards; this is the number the task is judged on (+200 = solved).
    score = 0
    observation, info = env.reset()
    num_iter = 0
    while not done and num_iter < MAX_ITER:
      # 1. Act (epsilon-greedy).
      action = agent.choose_action(observation)
      # 2. Observe the consequence.
      observation_, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
      score += reward
      # 3. Remember the transition.
      # Store `terminated`, not `done`: only a real end of episode (landed
      # or crashed) has no future value. A truncation (the 1000-step time
      # limit) cuts the episode short, but the lander still had a
      # future, so `learn` must keep bootstrapping from observation_.
      agent.store_transition(
        observation, action, reward, observation_, int(terminated))
      # 4. Learn from a random minibatch of past transitions.
      agent.learn()
      # 5. Move to the next state, so the next action is chosen from
      # where the lander is now.
      observation = observation_

      num_iter += 1
      if num_iter >= MAX_ITER:
        logger.warning(
          "The max iteration of the training loop of the agent is reached.")

    # How the episode ended. On the terminal step LunarLander adds +100
    # when the lander comes to rest and -100 when it crashes or flies off
    # screen, which outweighs the shaping part of that step's reward.
    # "timeout" (the 1000-step limit) with a score a bit below 0 means
    # the agent hovered instead of landing.
    if terminated:
      outcome = "landed" if reward > 0 else "crashed"
    elif truncated:
      outcome = "timeout"
    else:
      outcome = "max_iter"

    scores.append(score)
    # The average over the last 100 episodes is the standard measure of
    # progress: single scores vary a lot from one episode to the next.
    avg_score = np.mean(scores[-100:])
    logger.info(
      "episode: %d - score: %.1f - average score: %.1f - epsilon: %.2f"
      " - steps: %d - end: %s"
      % (i, score, avg_score, agent.epsilon, num_iter, outcome))
    eps_history.append(agent.epsilon)

    if avg_score > best_score:
      best_score = avg_score
      agent.save_models()

  # Close the rendering window and release the Box2D world.
  env.close()

  x = [i+1 for i in range(num_games)]
  plot_learning_curve(x, scores, eps_history, filename)
