"""Loaded countermovement-jump system-identification research package.

The package is deliberately thin at the plant boundary: the MuJoCo model,
contact mechanics, telemetry, and event definitions live in :mod:`plant` and
are wrapped by the public research modules below.
"""

from .model import build_model, compiled_model_signature, create_data
from .parameters import (
    default_parameters,
    load_parameters,
    parameter_schema,
    flatten_parameters,
    unflatten_parameters,
)
from .simulation import simulate_trial, trial_descriptor
from .validation import validate_parameters, validate_trial

__all__ = [
    "build_model",
    "compiled_model_signature",
    "create_data",
    "default_parameters",
    "load_parameters",
    "parameter_schema",
    "flatten_parameters",
    "unflatten_parameters",
    "simulate_trial",
    "trial_descriptor",
    "validate_parameters",
    "validate_trial",
]

