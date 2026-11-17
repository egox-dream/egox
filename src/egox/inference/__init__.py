"""Run a trained policy: load a run, prepare observations and execute action chunks.

Importing this package does not import torch; the model loads inside `MEDPolicy`.
Robot input/output and coordinates belong to `egox.robots`; collection and
rollout share `RobotClient`, with DDS state and separate camera acquisition.
"""
from .policy import MEDPolicy
from .rollout import rollout

__all__ = ['MEDPolicy', 'rollout']
