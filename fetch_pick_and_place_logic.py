from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Optional, Union
import threading
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

EventCallback = Callable[[dict[str, Any]], None]
SUPPORTED_METHODS = ("SAC", "TD3", "PPO")
SUPPORTED_REWARD_MODES = ("standard",)
FETCH_PICK_AND_PLACE_ENV_ID = "FetchPickAndPlace-v4"
FETCH_PICK_AND_PLACE_ENV_ID_FALLBACKS = ("FetchPickAndPlace-v4", "FetchPickAndPlace-v1")
FETCH_PICK_AND_PLACE_SOLVED_REWARD = 200.0
HALF_CHEETAH_ENV_ID = FETCH_PICK_AND_PLACE_ENV_ID
HALF_CHEETAH_SOLVED_REWARD = FETCH_PICK_AND_PLACE_SOLVED_REWARD
QUALITY_EVALUATION_EPISODES = 10


@lru_cache(maxsize=1)
def _verify_mujoco_runtime() -> None:
    pass


def _ensure_gymnasium_robotics() -> None:
    try:
        importlib.import_module("gymnasium_robotics")
    except Exception as exc:  # pragma: no cover - depends on installation state
        raise RuntimeError(
            "Fetch Pick And Place braucht das Paket 'gymnasium-robotics'. "
            "Installiere die Abhängigkeiten mit: pip install -r fetch_pick_and_place/requirements.txt"
        ) from exc


def make_fetch_env(render_mode: Optional[str] = None) -> gym.Env:
    _verify_mujoco_runtime()
    _ensure_gymnasium_robotics()
    last_error: Exception | None = None
    render_attempts: list[Optional[str]] = [render_mode]
    if render_mode == "rgb_array":
        render_attempts.append(None)
    elif render_mode is None:
        render_attempts = [None]

    for candidate_render_mode in render_attempts:
        for env_id in FETCH_PICK_AND_PLACE_ENV_ID_FALLBACKS:
            try:
                return gym.make(env_id, render_mode=candidate_render_mode)
            except ModuleNotFoundError as exc:  # pragma: no cover - depends on installation state
                if exc.name == "gymnasium_robotics" or "gymnasium_robotics" in str(exc):
                    _ensure_gymnasium_robotics()
                last_error = exc
            except Exception as exc:  # pragma: no cover - depends on OS/display backend
                last_error = exc

    message = (
        "MuJoCo konnte keinen gültigen Render-Kontext für Fetch Pick And Place erstellen. "
        "Das passiert typischerweise in headless-/Remote-Umgebungen, "
        "Windows-VMs oder ohne echte GUI-/OpenGL-Session. "
        "Starte die App auf einem lokalen Desktop oder deaktiviere die Live-Animation."
    )
    if last_error is not None:
        raise RuntimeError(message) from last_error
    raise RuntimeError(message)


def make_halfcheetah_env(render_mode: Optional[str] = None) -> gym.Env:
    return make_fetch_env(render_mode=render_mode)


def assess_training_quality(rewards: list[float], method: str) -> dict[str, Any]:
    finite = [float(r) for r in rewards if np.isfinite(r)]
    if not finite:
        return {
            "is_well_trained": False,
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "message": "Keine gültigen Evaluations-Rewards vorhanden.",
            "recommendations": ["Training erneut starten und Umgebung / Installation prüfen."],
        }

    mean_reward = float(np.mean(finite))
    std_reward = float(np.std(finite))
    if mean_reward >= HALF_CHEETAH_SOLVED_REWARD:
        return {
            "is_well_trained": True,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "message": "Das Modell ist erfolgreich trainiert.",
            "recommendations": [],
        }

    recommendations = ["total_timesteps erhöhen und danach erneut evaluieren."]
    if std_reward >= 1500.0:
        recommendations.extend([
            "learning_rate reduzieren und mehrere Seeds vergleichen.",
            "batch_size oder n_envs erhöhen, damit die Updates stabiler werden.",
        ])
        message = "Das Modell lernt, ist aber noch instabil."
    elif mean_reward < 0.0:
        recommendations.append("learning_rate und Netzwerkarchitektur im Parametervergleich prüfen.")
        if method == "PPO":
            recommendations.append("n_steps erhöhen und gae_lambda zwischen 0.90 und 0.98 vergleichen.")
        elif method == "SAC":
            recommendations.append("ent_coef='auto' verwenden und learning_starts erhöhen.")
        else:
            recommendations.append("learning_starts erhöhen und tau zwischen 0.003 und 0.01 vergleichen.")
        message = "Das Modell hat deutliche Lernprobleme."
    else:
        recommendations.append("learning_rate, Netzwerkarchitektur und batch_size im Parametervergleich testen.")
        if method == "PPO":
            recommendations.append("n_steps und gae_lambda gemeinsam vergleichen.")
        elif method == "SAC":
            recommendations.append("batch_size und ent_coef im Parametervergleich prüfen.")
        else:
            recommendations.append("buffer_size, learning_starts und policy_delay vergleichen.")
        message = "Das Modell verbessert sich, braucht aber weitere Trainingszeit."
    return {
        "is_well_trained": False,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "message": message,
        "recommendations": recommendations,
    }


class OnPolicyConfig:
    def __init__(
        self,
        method: str = "PPO",
        sweep_label: Optional[str] = None,
        env_id: str = FETCH_PICK_AND_PLACE_ENV_ID,
        reward_mode: str = "standard",
        total_timesteps: int = 1_000_000,
        training_stop_mode: str = "timesteps",
        target_episodes: Optional[int] = 5_000,
        seed: Optional[int] = 123,
        n_envs: int = 4,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        ent_coef: Union[float, str] = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        clip_range: float = 0.2,
        buffer_size: int = 300_000,
        learning_starts: int = 10_000,
        tau: float = 0.005,
        train_freq: int = 1,
        gradient_steps: int = 1,
        policy_delay: int = 2,
        net_arch_pi: str = "256,256",
        net_arch_vf: str = "256,256",
        activation_fn: str = "relu",
        ortho_init: bool = True,
        device: str = "auto",
        verbose: int = 0,
    ) -> None:
        self.method = method
        self.sweep_label = sweep_label
        self.env_id = env_id
        self.reward_mode = reward_mode
        self.total_timesteps = total_timesteps
        self.training_stop_mode = training_stop_mode
        self.target_episodes = target_episodes
        self.seed = seed
        self.n_envs = n_envs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.clip_range = clip_range
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.policy_delay = policy_delay
        self.net_arch_pi = net_arch_pi
        self.net_arch_vf = net_arch_vf
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self.device = device
        self.verbose = verbose

    def validate(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"Unterstützte Methoden: {', '.join(SUPPORTED_METHODS)}")
        if self.n_envs < 1:
            raise ValueError("n_envs muss mindestens 1 sein.")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda muss zwischen 0 und 1 liegen.")
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps muss positiv sein.")
        if self.training_stop_mode not in {"timesteps", "episodes"}:
            raise ValueError("training_stop_mode muss 'timesteps' oder 'episodes' sein.")
        if self.training_stop_mode == "episodes" and (self.target_episodes is None or self.target_episodes < 1):
            raise ValueError("target_episodes muss im Episodenmodus positiv sein.")


def get_default_parameters_for_method(method: str) -> dict[str, Any]:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unbekannte Methode: {method}")
    values = {
        "method": method,
        "env_id": FETCH_PICK_AND_PLACE_ENV_ID,
        "reward_mode": "standard",
        "total_timesteps": 1_000_000,
        "training_stop_mode": "timesteps",
        "target_episodes": 5_000,
        "seed": 123,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "clip_range": 0.2,
        "buffer_size": 300_000,
        "learning_starts": 10_000,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "policy_delay": 2,
        "net_arch_pi": "256,256",
        "net_arch_vf": "256,256",
        "activation_fn": "relu",
        "ortho_init": True,
        "device": "auto",
        "verbose": 0,
    }
    if method == "SAC":
        values.update(batch_size=256, ent_coef="auto", net_arch_pi="256,256", net_arch_vf="256,256")
    elif method == "TD3":
        values.update(learning_rate=1e-3, buffer_size=1_000_000, learning_starts=100, batch_size=100, gradient_steps=-1, net_arch_pi="400,300", net_arch_vf="400,300")
    else:
        values.update(net_arch_pi="256,256", net_arch_vf="256,256")
    return values


def parse_hidden_layers(value: str) -> list[int]:
    layers = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not layers or any(layer < 1 for layer in layers):
        raise ValueError("Netzwerkarchitektur muss positive Layergrößen enthalten.")
    return layers


def parse_architecture_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.replace("\n", ";").split(";") if item.strip()]
    for variant in variants:
        parse_hidden_layers(variant)
    return variants


def build_policy_kwargs(config: OnPolicyConfig) -> dict[str, Any]:
    activation_map = {
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "leakyrelu": torch.nn.LeakyReLU,
    }
    activation_name = config.activation_fn.strip().lower()
    if activation_name not in activation_map:
        allowed = ", ".join(sorted(activation_map))
        raise ValueError(f"Unbekannte Aktivierungsfunktion: {config.activation_fn}. Erlaubt: {allowed}")

    actor_layers = parse_hidden_layers(config.net_arch_pi)
    critic_layers = parse_hidden_layers(config.net_arch_vf)
    if config.method == "PPO":
        return {
            "net_arch": {"pi": actor_layers, "vf": critic_layers},
            "activation_fn": activation_map[activation_name],
            "ortho_init": config.ortho_init,
        }
    return {
        "net_arch": {"pi": actor_layers, "qf": critic_layers},
        "activation_fn": activation_map[activation_name],
    }


class SyncVectorEnvAdapter(VecEnv):
    def __init__(self, vector_env: gym.vector.SyncVectorEnv) -> None:
        self.vector_env = vector_env
        super().__init__(vector_env.num_envs, vector_env.single_observation_space, vector_env.single_action_space)
        self._actions: Any = None
        self._seeds: list[int | None] = [None for _ in range(self.num_envs)]
        self.reset_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

    def reset(self, *, seed: Optional[int | list[int] | tuple[int, ...] | np.ndarray] = None, options: Optional[dict[str, Any]] = None) -> np.ndarray:
        if seed is None:
            self._seeds = [None for _ in range(self.num_envs)]
        elif isinstance(seed, (list, tuple, np.ndarray)):
            self._seeds = [int(value) for value in seed]
        else:
            self._seeds = [int(seed) for _ in range(self.num_envs)]
        observations, infos = self.vector_env.reset(seed=self._seeds, options=options)
        self._seeds = [None for _ in range(self.num_envs)]
        self.reset_infos = self._split_infos(infos)
        return observations

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        observations, rewards, terminated, truncated, infos = self.vector_env.step(self._actions)
        dones = np.logical_or(terminated, truncated)
        split_infos = self._split_infos(infos)
        for index, done in enumerate(dones):
            if not done:
                continue
            split_infos[index]["TimeLimit.truncated"] = bool(truncated[index] and not terminated[index])
            if isinstance(observations, dict):
                split_infos[index]["terminal_observation"] = {
                    key: value[index] if hasattr(value, "__getitem__") else value
                    for key, value in observations.items()
                }
            else:
                split_infos[index]["terminal_observation"] = observations[index]
        return observations, rewards, dones, split_infos

    def close(self) -> None:
        self.vector_env.close()

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        target_indices = self._get_indices(indices)
        return [getattr(self.vector_env.envs[index], attr_name) for index in target_indices]

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        for index in self._get_indices(indices):
            setattr(self.vector_env.envs[index], attr_name, value)

    def env_method(self, method_name: str, *method_args: Any, indices: Any = None, **method_kwargs: Any) -> list[Any]:
        return [getattr(self.vector_env.envs[index], method_name)(*method_args, **method_kwargs) for index in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: Any = None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]

    def render(self, mode: Optional[str] = None) -> Any:
        return self.vector_env.render()

    def _split_infos(self, infos: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(infos, list):
            return [dict(item) if isinstance(item, dict) else {} for item in infos]

        result = [{} for _ in range(self.num_envs)]
        for key, value in infos.items():
            if key.startswith("_"):
                continue
            for index in range(self.num_envs):
                try:
                    if isinstance(value, dict):
                        nested: dict[str, Any] = {}
                        for nested_key, nested_value in value.items():
                            if isinstance(nested_value, (list, tuple, np.ndarray)) and len(nested_value) == self.num_envs:
                                nested[nested_key] = nested_value[index]
                            else:
                                nested[nested_key] = nested_value
                        result[index][key] = nested
                    else:
                        result[index][key] = value[index]
                except (IndexError, TypeError, KeyError):
                    pass
        return result


class TrainingCallback(BaseCallback):
    def __init__(
        self,
        stop_event: threading.Event,
        event_callback: Optional[EventCallback] = None,
        *,
        total_timesteps: int = 1,
        method: str = "PPO",
        event_interval: int = 64,
        training_stop_mode: str = "timesteps",
        target_episodes: Optional[int] = None,
        pause_event: Optional[threading.Event] = None,
        render_during_training: bool = False,
        frame_capture_steps: int = 64,
    ) -> None:
        super().__init__(verbose=0)
        self.stop_event = stop_event
        self.event_callback = event_callback
        self.total_timesteps = max(1, int(total_timesteps))
        self.method = method
        self.event_interval = max(1, int(event_interval))
        self.training_stop_mode = training_stop_mode
        self.target_episodes = target_episodes
        self.pause_event = pause_event
        self.render_during_training = bool(render_during_training)
        self.frame_capture_steps = max(1, int(frame_capture_steps))
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self._running_rewards: dict[int, float] = {}
        self._running_lengths: dict[int, int] = {}

    def _emit(self, event: dict[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(event)

    def _render_frame(self) -> Any:
        if not self.render_during_training:
            return None
        try:
            frame = self.training_env.render()
        except Exception:
            return None
        if isinstance(frame, (list, tuple)):
            return frame[0] if frame else None
        if isinstance(frame, np.ndarray) and frame.ndim == 4:
            return frame[0]
        return frame

    def _on_step(self) -> bool:
        while self.pause_event is not None and self.pause_event.is_set():
            if self.stop_event.is_set():
                self._emit({"type": "cancelled", "method": self.method})
                return False
            time.sleep(0.05)
        if self.stop_event.is_set():
            self._emit({"type": "cancelled", "method": self.method})
            return False

        rewards = np.asarray(self.locals.get("rewards", []), dtype=float)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool)
        infos = self.locals.get("infos", [])
        for index, reward in enumerate(rewards):
            self._running_rewards[index] = self._running_rewards.get(index, 0.0) + float(reward)
            self._running_lengths[index] = self._running_lengths.get(index, 0) + 1
            if index >= len(dones) or not dones[index]:
                continue
            info = infos[index] if index < len(infos) and isinstance(infos[index], dict) else {}
            episode = info.get("episode", {})
            episode_reward = float(episode.get("r", self._running_rewards[index]))
            episode_length = int(episode.get("l", self._running_lengths[index]))
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self._running_rewards[index] = 0.0
            self._running_lengths[index] = 0
            self._emit({
                "type": "episode",
                "method": self.method,
                "reward": episode_reward,
                "length": episode_length,
                "episodes": len(self.episode_rewards),
            })
            if self.training_stop_mode == "episodes" and self.target_episodes is not None and len(self.episode_rewards) >= self.target_episodes:
                self._emit({
                    "type": "progress",
                    "method": self.method,
                    "timesteps": int(self.num_timesteps),
                    "total_timesteps": self.total_timesteps,
                    "ratio": 1.0,
                })
                return False

        if self.num_timesteps % self.event_interval == 0:
            self._emit({
                "type": "progress",
                "method": self.method,
                "timesteps": int(self.num_timesteps),
                "total_timesteps": self.total_timesteps,
                "ratio": min(1.0, float(self.num_timesteps) / self.total_timesteps),
            })
        if self.num_timesteps % self.frame_capture_steps == 0:
            frame = self._render_frame()
            if frame is not None:
                self._emit({"type": "frame", "method": self.method, "frame": frame})
        return True


class FetchPickAndPlaceTrainer:
    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.configs: dict[str, OnPolicyConfig] = {}

    @staticmethod
    def _make_env(config: OnPolicyConfig, render_mode: Optional[str] = None) -> gym.Env:
        _verify_mujoco_runtime()
        _ensure_gymnasium_robotics()
        try:
            return gym.make(config.env_id, render_mode=render_mode)
        except Exception as exc:  # pragma: no cover - depends on OS/display backend
            if render_mode == "rgb_array":
                try:
                    return gym.make(config.env_id, render_mode=None)
                except Exception:
                    pass
            message = (
                "MuJoCo konnte keinen gültigen Render-Kontext erstellen. "
                "Das passiert typischerweise in headless-/Remote-Umgebungen, "
                "Windows-VMs oder ohne echte GUI-/OpenGL-Session. "
                "Starte die App auf einem lokalen Desktop oder deaktiviere die Live-Animation."
            )
            raise RuntimeError(message) from exc

    @classmethod
    def _make_vec_env(cls, config: OnPolicyConfig, render_during_training: bool = False):
        render_mode = "rgb_array" if render_during_training else None
        factories = [
            (lambda r=render_mode: cls._make_env(config, r))
            for _ in range(config.n_envs)
        ]
        vector_env = gym.vector.SyncVectorEnv(factories)
        environment = SyncVectorEnvAdapter(vector_env)
        if config.seed is not None:
            environment.reset(seed=config.seed)
        return environment

    @staticmethod
    def _create_model(config: OnPolicyConfig, environment: VecEnv):
        policy_name = "MultiInputPolicy" if isinstance(environment.observation_space, gym.spaces.Dict) else "MlpPolicy"
        common = {
            "policy": policy_name,
            "env": environment,
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "batch_size": config.batch_size,
            "policy_kwargs": build_policy_kwargs(config),
            "seed": config.seed,
            "device": config.device,
            "verbose": config.verbose,
        }
        if config.method == "PPO":
            return PPO(
                n_steps=config.n_steps,
                n_epochs=config.n_epochs,
                gae_lambda=config.gae_lambda,
                ent_coef=float(config.ent_coef),
                vf_coef=config.vf_coef,
                max_grad_norm=config.max_grad_norm,
                clip_range=config.clip_range,
                **common,
            )
        off_policy = {**common, "buffer_size": config.buffer_size, "learning_starts": config.learning_starts, "tau": config.tau, "train_freq": config.train_freq, "gradient_steps": config.gradient_steps}
        if config.method == "SAC":
            return SAC(ent_coef=config.ent_coef, **off_policy)
        return TD3(policy_delay=config.policy_delay, **off_policy)

    def train(
        self,
        config: OnPolicyConfig,
        stop_event: threading.Event,
        event_callback: Optional[EventCallback] = None,
        render_during_training: bool = False,
        frame_capture_steps: int = 64,
        pause_event: Optional[threading.Event] = None,
        auto_save: bool = True,
    ) -> dict[str, Any]:
        config.validate()
        learn_timesteps = int(config.total_timesteps)
        if config.training_stop_mode == "episodes" and config.target_episodes is not None:
            learn_timesteps = max(learn_timesteps, int(config.target_episodes) * 2000)
        environment = self._make_vec_env(config, render_during_training)
        callback = TrainingCallback(
            stop_event,
            event_callback,
            total_timesteps=learn_timesteps,
            method=config.method,
            training_stop_mode=config.training_stop_mode,
            target_episodes=config.target_episodes,
            pause_event=pause_event,
            render_during_training=render_during_training,
            frame_capture_steps=frame_capture_steps,
        )
        model = self._create_model(config, environment)
        self.models[config.method] = model
        self.configs[config.method] = config
        model.learn(total_timesteps=learn_timesteps, callback=callback, progress_bar=False)
        rewards = callback.episode_rewards
        result = {
            "method": config.method,
            "rewards": rewards,
            "lengths": callback.episode_lengths,
            "cancelled": stop_event.is_set(),
            "quality": assess_training_quality(rewards, config.method),
            "auto_saved_path": None,
        }
        if auto_save:
            try:
                path = Path(__file__).resolve().parent / "saved_models" / f"{config.method.lower()}_fetch_pick_and_place_model.zip"
                path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(path))
                result["auto_saved_path"] = str(path)
            except Exception:
                result["auto_saved_path"] = None
        return result

    def evaluate(self, model: Any, episodes: int = 10, seed: Optional[int] = None) -> list[float]:
        rewards: list[float] = []
        env = make_halfcheetah_env()
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode if seed is not None else None)
            total_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
            rewards.append(total_reward)
        env.close()
        return rewards

    def save_model(self, method: str, path: str | Path) -> str:
        model = self.models.get(method)
        if model is None:
            raise ValueError(f"Kein geladenes Modell für {method} vorhanden.")
        full_path = str(path)
        model.save(full_path)
        return full_path

    def load_model(self, method: str, path: str | Path) -> Any:
        env = make_halfcheetah_env()
        if method == "PPO":
            model = PPO.load(str(path), env=env)
        elif method == "SAC":
            model = SAC.load(str(path), env=env)
        else:
            model = TD3.load(str(path), env=env)
        self.models[method] = model
        self.configs[method] = OnPolicyConfig(method=method)
        env.close()
        return model


HalfCheetahTrainer = FetchPickAndPlaceTrainer
Baselines3Trainer = FetchPickAndPlaceTrainer
GymnasiumVectorEnvAdapter = SyncVectorEnvAdapter


__all__ = [
    "EventCallback",
    "SUPPORTED_METHODS",
    "SUPPORTED_REWARD_MODES",
    "FETCH_PICK_AND_PLACE_ENV_ID",
    "FETCH_PICK_AND_PLACE_SOLVED_REWARD",
    "HALF_CHEETAH_ENV_ID",
    "HALF_CHEETAH_SOLVED_REWARD",
    "OnPolicyConfig",
    "get_default_parameters_for_method",
    "make_fetch_env",
    "make_halfcheetah_env",
    "FetchPickAndPlaceTrainer",
    "HalfCheetahTrainer",
    "Baselines3Trainer",
    "SyncVectorEnvAdapter",
    "GymnasiumVectorEnvAdapter",
    "assess_training_quality",
    "parse_hidden_layers",
    "parse_architecture_variants",
]
