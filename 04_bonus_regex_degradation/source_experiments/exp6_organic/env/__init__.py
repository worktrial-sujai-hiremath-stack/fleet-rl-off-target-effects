"""fix_bug env package marker.

When installed into SkyRL, this sits at
`SkyRL/skyrl-gym/skyrl_gym/envs/fix_bug/__init__.py`. The env is registered
by appending a `register(id="fix_bug", ...)` call to the parent
`skyrl_gym/envs/__init__.py` in the launcher's setup block.
"""
