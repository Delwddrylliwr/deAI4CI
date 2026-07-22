# -*- coding: utf-8 -*-
"""Config serialization: NaturalCascadeConfig / NCPSimConfig <-> JSON-safe dicts.

JSON has no numpy array or tuple types, so three fields need special handling:
  theta_A, theta_B   : Optional[np.ndarray] -> null or [float, ...]
  per_leaf_loss_params: Optional[List[Tuple[float,float]]] -> null or [[a,b],...]
"""
from typing import Any, Dict, Optional

import numpy as np


def config_to_dict(config) -> Dict[str, Any]:
    """Serialize NaturalCascadeConfig or NCPSimConfig to a JSON-safe dict."""
    d = {}
    for field_name, field_val in config.__dataclass_fields__.items():
        val = getattr(config, field_name)
        if isinstance(val, np.ndarray):
            d[field_name] = val.tolist()
        elif val is None:
            d[field_name] = None
        elif isinstance(val, (list, tuple)) and val and isinstance(val[0], (list, tuple)):
            # per_leaf_loss_params: List[Tuple[float,float]]
            d[field_name] = [list(pair) for pair in val]
        else:
            d[field_name] = val
    return d


def dict_to_nc_config(d: Dict[str, Any]):
    """Reconstruct NaturalCascadeConfig from a dict produced by config_to_dict."""
    from analysis.natural_cascade import NaturalCascadeConfig

    kwargs = dict(d)
    for key in ("theta_A", "theta_B"):
        if kwargs.get(key) is not None:
            kwargs[key] = np.array(kwargs[key], dtype=np.float32)
    if kwargs.get("per_leaf_loss_params") is not None:
        kwargs["per_leaf_loss_params"] = [tuple(pair) for pair in kwargs["per_leaf_loss_params"]]
    return NaturalCascadeConfig(**kwargs)


def dict_to_ncp_config(d: Dict[str, Any]):
    """Reconstruct NCPSimConfig from a dict produced by config_to_dict."""
    from analysis.ncp_runner import NCPSimConfig

    kwargs = dict(d)
    for key in ("theta_A", "theta_B"):
        if kwargs.get(key) is not None:
            kwargs[key] = np.array(kwargs[key], dtype=np.float32)
    return NCPSimConfig(**kwargs)


def dict_to_generic_config(d: Dict[str, Any]):
    """Reconstruct GenericTopologyConfig from a dict produced by config_to_dict."""
    from analysis.generic_topology_runner import GenericTopologyConfig

    kwargs = dict(d)
    for key in ("theta_A", "theta_B"):
        if kwargs.get(key) is not None:
            kwargs[key] = np.array(kwargs[key], dtype=np.float32)
    return GenericTopologyConfig(**kwargs)


def task_id_from_config(config) -> str:
    """Stable filesystem-safe task ID matching run_timesep.py's cache-key formula."""
    return config.name.replace("/", "__").replace("=", "")
