"""h2_tool_edit env package (Exp 8).

When installed into SkyRL this sits at
`SkyRL/skyrl-gym/skyrl_gym/envs/h2_tool_edit/__init__.py`. Registration of
`id="h2_tool_edit"` is appended to the parent `skyrl_gym/envs/__init__.py`
by install.sh (same pattern as fix_bug).
"""

from skyrl_gym.envs.registration import register

try:
    from skyrl_gym.envs.h2_tool_edit.env import H2ToolEditEnv  # noqa: F401
except ImportError:  # local tests without install
    H2ToolEditEnv = None  # type: ignore

try:
    register(
        id="h2_tool_edit",
        entry_point="skyrl_gym.envs.h2_tool_edit.env:H2ToolEditEnv",
    )
except Exception:
    # Already registered or registry not ready in local-test import.
    pass
