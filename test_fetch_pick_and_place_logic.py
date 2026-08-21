import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


def test_fetch_pick_and_place_logic_imports():
    module_path = Path(__file__).with_name("fetch_pick_and_place_logic.py")
    spec = spec_from_file_location("fetch_pick_and_place_logic", module_path)
    assert spec is not None and spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.FETCH_PICK_AND_PLACE_ENV_ID in {"FetchPickAndPlace-v4", "FetchPickAndPlace-v1"}
    assert "SAC" in module.SUPPORTED_METHODS
    assert "TD3" in module.SUPPORTED_METHODS
    assert "PPO" in module.SUPPORTED_METHODS


def test_fetch_pick_and_place_gui_imports():
    module_dir = Path(__file__).resolve().parent
    for path in (str(module_dir), str(module_dir.parent)):
        if path not in sys.path:
            sys.path.insert(0, path)

    module_path = module_dir / "fetch_pick_and_place_gui.py"
    spec = spec_from_file_location("fetch_pick_and_place_gui", module_path)
    assert spec is not None and spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "FetchPickAndPlaceUI")


def test_fetch_pick_and_place_supports_dense_reward_and_her_methods():
    module_path = Path(__file__).with_name("fetch_pick_and_place_logic.py")
    spec = spec_from_file_location("fetch_pick_and_place_logic", module_path)
    assert spec is not None and spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    assert {"standard", "dense", "shaped"}.issubset(set(module.SUPPORTED_REWARD_MODES))
    assert "HER-SAC" in module.SUPPORTED_METHODS
    assert "HER-TD3" in module.SUPPORTED_METHODS


def test_her_replay_buffer_kwargs_match_sb3_api(monkeypatch):
    module_path = Path(__file__).with_name("fetch_pick_and_place_logic.py")
    spec = spec_from_file_location("fetch_pick_and_place_logic", module_path)
    assert spec is not None and spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    seen = {}

    class FakeSAC:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(module, "SAC", FakeSAC)
    config = module.OnPolicyConfig(method="HER-SAC", reward_mode="dense")

    class FakeEnv:
        observation_space = module.gym.spaces.Dict({
            "observation": module.gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "desired_goal": module.gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "achieved_goal": module.gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        })
        action_space = module.gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    model = module.FetchPickAndPlaceTrainer._create_model(config, FakeEnv())

    assert model is not None
    assert "replay_buffer_class" in seen
    assert "replay_buffer_kwargs" in seen
    assert "online_sampling" not in seen["replay_buffer_kwargs"]
    assert seen["replay_buffer_kwargs"]["goal_selection_strategy"] == "future"


def test_make_fetch_env_initializes_robotics_registration(monkeypatch):
    module_path = Path(__file__).with_name("fetch_pick_and_place_logic.py")
    spec = spec_from_file_location("fetch_pick_and_place_logic", module_path)
    assert spec is not None and spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    calls = []

    def fake_ensure():
        calls.append("ensure")

    def fake_make(env_id, render_mode=None):
        calls.append(("make", env_id, render_mode))
        return object()

    monkeypatch.setattr(module, "_ensure_gymnasium_robotics", fake_ensure)
    monkeypatch.setattr(module.gym, "make", fake_make)

    env = module.make_fetch_env(render_mode="rgb_array")

    assert env is not None
    assert calls[0] == "ensure"
    assert calls[1][0] == "make"


def test_sync_vector_env_adapter_handles_dict_observations():
    module_path = Path(__file__).with_name("fetch_pick_and_place_logic.py")
    spec = spec_from_file_location("fetch_pick_and_place_logic", module_path)
    assert spec is not None and spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeVecEnv:
        num_envs = 1
        single_observation_space = module.gym.spaces.Dict({
            "observation": module.gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float),
        })
        single_action_space = module.gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)

        def __init__(self):
            self.envs = [object()]

        def step(self, actions):
            return (
                {"observation": np.array([[0.1, 0.2]])},
                np.array([1.0]),
                np.array([True]),
                np.array([False]),
                {"episode": {"r": np.array([1.0]), "l": np.array([2])}},
            )

        def reset(self, seed=None, options=None):
            return {"observation": np.array([[0.0, 0.0]])}, {}

    import numpy as np

    adapter = module.SyncVectorEnvAdapter(FakeVecEnv())
    obs, rewards, dones, infos = adapter.step_wait()

    assert obs["observation"][0].tolist() == [0.1, 0.2]
    assert rewards[0] == 1.0
    assert bool(dones[0]) is True
    assert infos[0]["terminal_observation"]["observation"].tolist() == [0.1, 0.2]


def test_make_vec_env_uses_same_render_mode_for_every_env(monkeypatch):
    module_path = Path(__file__).with_name("fetch_pick_and_place_logic.py")
    spec = spec_from_file_location("fetch_pick_and_place_logic", module_path)
    assert spec is not None and spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    seen = []

    class FakeSyncVectorEnv:
        def __init__(self, env_fns):
            self.num_envs = len(env_fns)
            self.single_observation_space = module.gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
            self.single_action_space = module.gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
            self.envs = [fn() for fn in env_fns]

        def reset(self, seed=None, options=None):
                return np.zeros((self.num_envs, 2)), {"episode": {"r": np.zeros(self.num_envs), "l": np.zeros(self.num_envs, dtype=int)}}
        def step(self, actions):
            return np.zeros((self.num_envs, 2)), np.zeros(self.num_envs), np.zeros(self.num_envs, dtype=bool), np.zeros(self.num_envs, dtype=bool), [{} for _ in range(self.num_envs)]

        def close(self):
            pass

    def fake_make_env(config, render_mode=None):
        seen.append(render_mode)
        return object()

    monkeypatch.setattr(module.gym.vector, "SyncVectorEnv", FakeSyncVectorEnv)
    monkeypatch.setattr(module.FetchPickAndPlaceTrainer, "_make_env", staticmethod(fake_make_env))

    config = module.OnPolicyConfig(method="PPO", n_envs=3)
    env = module.FetchPickAndPlaceTrainer._make_vec_env(config, render_during_training=True)

    assert env is not None
    assert seen == ["rgb_array", "rgb_array", "rgb_array"]


def test_make_fetch_env_falls_back_without_render_context(monkeypatch):
    module_path = Path(__file__).with_name("fetch_pick_and_place_logic.py")
    spec = spec_from_file_location("fetch_pick_and_place_logic", module_path)
    assert spec is not None and spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    calls = []

    def fake_make(env_id, render_mode=None):
        calls.append(render_mode)
        if render_mode == "rgb_array":
            raise RuntimeError("invalid render context")
        return object()

    monkeypatch.setattr(module.gym, "make", fake_make)

    env = module.make_fetch_env(render_mode="rgb_array")

    assert env is not None
    assert calls[0] == "rgb_array"
    assert calls[-1] is None
    assert len(calls) == 3


def test_sync_vector_env_adapter_delegates_compute_reward_to_unwrapped_env():
    module_path = Path(__file__).with_name("fetch_pick_and_place_logic.py")
    spec = spec_from_file_location("fetch_pick_and_place_logic", module_path)
    assert spec is not None and spec.loader is not None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeWrappedEnv:
        def __init__(self):
            self.unwrapped = self
            self.compute_reward_called = False

        def compute_reward(self, achieved_goal, desired_goal, info):
            self.compute_reward_called = True
            return 7.5

    class FakeVecEnv:
        num_envs = 1
        single_observation_space = module.gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
        single_action_space = module.gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)

        def __init__(self):
            self.envs = [FakeWrappedEnv()]

        def reset(self, seed=None, options=None):
            return np.zeros((1, 2)), {}

        def step(self, actions):
            return np.zeros((1, 2)), np.zeros(1), np.zeros(1, dtype=bool), np.zeros(1, dtype=bool), [{}]

        def close(self):
            pass

    adapter = module.SyncVectorEnvAdapter(FakeVecEnv())
    result = adapter.env_method("compute_reward", np.array([1.0, 2.0]), np.array([3.0, 4.0]), {"info": 1}, indices=[0])

    assert result == [7.5]
    assert adapter.vector_env.envs[0].compute_reward_called is True
