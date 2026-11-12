"""Run a trained policy: load a run, prepare observations and execute action chunks.

Importing this package does not import torch; the model loads inside `MXTPolicy`.
Robot input/output and coordinates belong to `egox.robots`; collection and
rollout connect through its shared DDS `RobotClient`.
"""
from .policy import MXTPolicy
from .rollout import rollout

__all__ = ['MXTPolicy', 'rollout']
