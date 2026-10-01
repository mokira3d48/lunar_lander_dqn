"""Run a trained Dueling DQN agent on LunarLander-v3.

Usage:

  python run_dualing_dqn_torch.py path/to/dir [--episodes N] [--no-render]

`path/to/dir` is either the folder of one training run (it holds
`q_eval_dueling_dqn`), e.g. `tmp/dqn/20261001-050330-1208731`, or a folder
of runs such as `tmp/dqn`, in which case the most recently saved checkpoint
among its runs is loaded.
"""
import argparse
import os
import sys
import time as tm
from typing import Tuple

from loguru import logger
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Logging: messages go to the terminal and are appended to `./logs`.
# Every line of `./logs` carries the id of the run that wrote it, so the
# episodes of runs launched at the same time can be told apart in the file.
RUN_ID = tm.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
LOG_FORMAT = (
  "{time:YYYY-MM-DD HH:mm:ss} [{level}] {extra[name]} [{extra[run]}]: {message}")
logger.remove()
logger.configure(extra={"name": "Dualing DQN", "run": RUN_ID})
logger.add(sys.stdout, level="INFO", format="{level}: {message}")
logger.add('./run_dualing_dqn_torch.log', level="INFO", format=LOG_FORMAT)

CHECKPOINT_NAME = 'q_eval_dueling_dqn'


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

  def load_checkpoint(self):
    """Load the weights saved by `save_checkpoint`.

    `map_location` lets a checkpoint saved on a GPU be loaded on a
    CPU-only machine (and vice versa).
    """
    logger.info("... loading checkpoint ...")
    self.load_state_dict(
      torch.load(self.checkpoint_file, map_location=self.device))


def find_checkpoint_dir(path: str) -> str:
  """Return the folder holding the last saved `q_eval` checkpoint."""
  if os.path.isfile(os.path.join(path, CHECKPOINT_NAME)):
    return path

  candidates = [
    os.path.join(path, name) for name in os.listdir(path)
    if os.path.isfile(os.path.join(path, name, CHECKPOINT_NAME))
  ]
  if not candidates:
    raise FileNotFoundError(
      f"no '{CHECKPOINT_NAME}' checkpoint in {path} or its subfolders")
  return max(
    candidates,
    key=lambda d: os.path.getmtime(os.path.join(d, CHECKPOINT_NAME)))


def choose_action(model: Model, observation: np.ndarray) -> int:
  """Greedy action: argmax of the advantage A(s, a), no exploration."""
  state = torch.tensor(observation[np.newaxis, :]).to(model.device)
  with torch.no_grad():
    _, advantage = model.forward(state)
  return torch.argmax(advantage).item()


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument(
    'checkpoint_dir',
    help="folder of a training run, or a folder of runs (latest is used)")
  parser.add_argument(
    '--episodes', type=int, default=5, help="number of episodes to play")
  parser.add_argument(
    '--no-render', action='store_true', help="play without a window")
  args = parser.parse_args()

  if not os.path.isdir(args.checkpoint_dir):
    parser.error(f"{args.checkpoint_dir} is not a directory")
  try:
    chkpt_dir = find_checkpoint_dir(args.checkpoint_dir)
  except FileNotFoundError as e:
    logger.error(str(e))
    return 1

  env = gym.make(
    id='LunarLander-v3',
    render_mode=None if args.no_render else 'human'
  )
  logger.info("Input dims: " + str(env.observation_space.shape[0]))
  model = Model(
    alpha=5e-4,  # unused: no training here, but required by Model
    n_actions=env.action_space.n,
    name='q_eval',
    input_dims=env.observation_space.shape[0],
    chkpt_dir=chkpt_dir,
  )
  logger.info("checkpoint: %s" % model.checkpoint_file)
  model.load_checkpoint()
  model.eval()

  scores = []
  for i in range(args.episodes):
    observation, _ = env.reset()
    done = False
    score = 0.0
    steps = 0
    while not done:
      action = choose_action(model, observation)
      observation, reward, terminated, truncated, _ = env.step(action)
      done = terminated or truncated
      score += reward
      steps += 1

    # Same rule as in training: the terminal step carries +100 when the
    # lander comes to rest and -100 when it crashes.
    if terminated:
      outcome = "landed" if reward > 0 else "crashed"
    else:
      outcome = "timeout"
    scores.append(score)
    logger.info(
      "episode: %d - score: %.1f - steps: %d - end: %s"
      % (i, score, steps, outcome))

  env.close()
  logger.info(
    "average score over %d episodes: %.1f" % (len(scores), np.mean(scores)))
  return 0


if __name__ == '__main__':
  sys.exit(main())
