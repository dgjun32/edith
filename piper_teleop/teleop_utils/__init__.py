"""Teleoperation-specific setup, kept out of robot and policy code."""

from .gello import apply_gello_overrides, validate_gello

__all__ = ["apply_gello_overrides", "validate_gello"]
