"""Torch MADDPG-style training for the B5G offline replay environment.

The bundled B5G release gives one labelled configuration per sample, so each
``B5GSlicingEnv`` episode is a contextual bandit step rather than a multi-step
online rollout. This trainer keeps the MADDPG actor/centralized-critic shape,
but uses the observed reward as the one-step target until a live simulator is
available.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

try:
    from .b5g_env import SLICE_TYPES, AgentObservation, B5GSlicingEnv
except ImportError:
    from b5g_env import SLICE_TYPES, AgentObservation, B5GSlicingEnv

OBSERVATION_DIM = 13
ACTION_DIM = len(SLICE_TYPES)


class Actor(nn.Module):
    """Shared policy that maps one agent observation to slice shares."""

    def __init__(self, observation_dim: int = OBSERVATION_DIM, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.network(observation), dim=-1)


class CentralizedCritic(nn.Module):
    """Critic over one agent's state and the joint slice allocation."""

    def __init__(self, observation_dim: int = OBSERVATION_DIM, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim + ACTION_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((observation, action), dim=-1)).squeeze(-1)


@dataclass
class TrainingConfig:
    episodes: int = 20
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 2e-3
    discount: float = 0.99
    exploration_noise: float = 0.05
    seed: int = 7
    checkpoint_path: str | None = None


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.items: list[tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, bool]] = []

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_observation: torch.Tensor | None = None,
        done: bool = True,
    ) -> None:
        if len(self.items) >= self.capacity:
            self.items.pop(0)
        if next_observation is None:
            next_observation = observation
        self.items.append((observation.detach(), action.detach(), float(reward), next_observation.detach(), done))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.items, min(batch_size, len(self.items)))
        observations, actions, rewards, next_observations, dones = zip(*batch)
        return (
            torch.stack(observations),
            torch.stack(actions),
            torch.tensor(rewards),
            torch.stack(next_observations),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.items)


class MADDPGTrainer:
    """Train a shared MADDPG actor against the offline B5G reward signal."""

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        self.actor = Actor()
        self.critic = CentralizedCritic()
        self.target_actor = Actor()
        self.target_critic = CentralizedCritic()
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate
        )
        self.replay = ReplayBuffer()

    @staticmethod
    def _state(observation: AgentObservation) -> torch.Tensor:
        values = observation.to_vector()
        if len(values) != OBSERVATION_DIM:
            raise ValueError(f"Expected {OBSERVATION_DIM} observation values, got {len(values)}")
        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _action_dict(action: torch.Tensor) -> dict[str, float]:
        return {slice_type: float(action[index]) for index, slice_type in enumerate(SLICE_TYPES)}

    def select_action(self, observation: AgentObservation, explore: bool = False) -> torch.Tensor:
        with torch.no_grad():
            action = self.actor(self._state(observation).unsqueeze(0)).squeeze(0)
        if explore:
            action = (action + torch.randn_like(action) * self.config.exploration_noise).clamp(0.0, 1.0)
            action = action / action.sum().clamp(min=1e-6)
        return action

    def select_actions(
        self, observations: dict[str, AgentObservation], explore: bool = False
    ) -> dict[str, dict[str, float]]:
        if not observations:
            return {}
        states = torch.stack([self._state(observation) for observation in observations.values()])
        with torch.no_grad():
            actions = self.actor(states)
        if explore:
            actions = (actions + torch.randn_like(actions) * self.config.exploration_noise).clamp(0.0, 1.0)
            actions = actions / actions.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return {
            agent_id: self._action_dict(action)
            for agent_id, action in zip(observations, actions)
        }

    def predict_reward(self, observation: AgentObservation, action: dict[str, float]) -> float:
        state = self._state(observation).unsqueeze(0)
        action_tensor = torch.tensor(
            [[action.get(slice_type, 0.0) for slice_type in SLICE_TYPES]],
            dtype=torch.float32,
        )
        with torch.no_grad():
            return float(self.critic(state, action_tensor).item())

    def _update(self, batch_size: int = 256) -> None:
        if len(self.replay) == 0:
            return
        observations, actions, rewards, next_observations, dones = self.replay.sample(batch_size)
        with torch.no_grad():
            next_actions = self.target_actor(next_observations)
            targets = rewards + self.config.discount * self.target_critic(next_observations, next_actions) * (1.0 - dones)
        critic_loss = nn.functional.mse_loss(self.critic(observations, actions), targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_actions = self.actor(observations)
        actor_loss = -self.critic(observations, actor_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def train_episode(self, env: B5GSlicingEnv) -> float:
        observations = env.reset()
        actions = self.select_actions(observations, explore=True)
        _, rewards, _, _ = env.step(actions)
        for agent_id, observation in observations.items():
            self.replay.add(
                self._state(observation),
                torch.tensor([actions[agent_id][kind] for kind in SLICE_TYPES]),
                rewards[agent_id],
                done=True,
            )
        self._update()
        return sum(rewards.values()) / max(len(rewards), 1)

    def train_multistep(self, env, episodes: int = 1) -> list[float]:
        """Train over an environment exposing sequential reset/step transitions."""
        history: list[float] = []
        for _ in range(episodes):
            observations = env.reset()
            episode_rewards: list[float] = []
            while observations:
                actions = self.select_actions(observations, explore=True)
                next_observations, rewards, dones, _ = env.step(actions)
                for agent_id, observation in observations.items():
                    self.replay.add(
                        self._state(observation),
                        torch.tensor([actions[agent_id][kind] for kind in SLICE_TYPES]),
                        rewards[agent_id],
                        self._state(next_observations[agent_id]) if agent_id in next_observations else None,
                        dones[agent_id],
                    )
                    episode_rewards.append(rewards[agent_id])
                self._update()
                observations = next_observations
            history.append(sum(episode_rewards) / max(len(episode_rewards), 1))
        return history

    def train(self, env: B5GSlicingEnv) -> list[float]:
        history = [self.train_episode(env) for _ in range(self.config.episodes)]
        if self.config.checkpoint_path:
            self.save(self.config.checkpoint_path)
        return history

    def evaluate(self, env: B5GSlicingEnv, episodes: int = 1) -> dict[str, float]:
        rewards: list[float] = []
        sla_violations = 0
        total_agents = 0
        inference_start = time.perf_counter()
        for _ in range(episodes):
            observations = env.reset()
            actions = self.select_actions(observations)
            inference_start = time.perf_counter()
            _, episode_rewards, _, infos = env.step(actions)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            rewards.append(sum(episode_rewards.values()) / max(len(episode_rewards), 1))
            total_agents += len(observations)
            sla_violations += sum(
                1 for info in infos.values()
                if info["reward_breakdown"].sla_violation_penalty < 0.0
            )
        return {
            "episodes": float(episodes),
            "mean_reward": sum(rewards) / len(rewards),
            "sla_violation_rate": sla_violations / max(total_agents, 1),
            "inference_ms_per_episode": inference_ms,
            "inference_ms_per_agent": inference_ms / max(len(observations), 1),
        }

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "config": self.config},
            destination,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())


def train_from_command_line(env: B5GSlicingEnv, episodes: int = 20) -> Iterable[float]:
    """Small entry point for reproducible local smoke training."""
    return MADDPGTrainer(TrainingConfig(episodes=episodes)).train(env)


if __name__ == "__main__":
    env = B5GSlicingEnv()
    trainer = MADDPGTrainer(TrainingConfig(episodes=3))
    history = trainer.train(env)
    print(f"episodes={len(history)} mean_train_reward={sum(history) / len(history):.4f}")
    print(trainer.evaluate(env))
