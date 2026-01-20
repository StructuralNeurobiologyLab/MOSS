#!/usr/bin/env python3
"""
Model aggregation strategies for federated learning.

Implements FedAvg and related aggregation algorithms.
"""

import copy
from typing import Dict, List, Tuple, Optional

import torch


class FedAvgAggregator:
    """
    Federated Averaging (FedAvg) aggregation.

    Computes a weighted average of model weights from multiple clients.
    Each client's contribution is weighted by their number of training samples.

    Reference:
        McMahan et al., "Communication-Efficient Learning of Deep Networks
        from Decentralized Data" (2017)
    """

    def __init__(self):
        """Initialize the aggregator."""
        self.pending_updates: List[Tuple[dict, float]] = []  # (weights, contribution_weight)
        self.global_weights: Optional[dict] = None
        self.aggregation_count: int = 0

    def add_update(self, weights: dict, contribution_weight: float = 1.0) -> None:
        """
        Add a weight update from a client.

        Args:
            weights: PyTorch model state dict
            contribution_weight: Weight for this contribution (e.g., num_samples)
        """
        # Deep copy to avoid reference issues
        weights_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v
                       for k, v in weights.items()}
        self.pending_updates.append((weights_copy, contribution_weight))

    def clear_updates(self) -> None:
        """Clear all pending updates."""
        self.pending_updates = []

    @property
    def update_count(self) -> int:
        """Get the number of pending updates."""
        return len(self.pending_updates)

    def aggregate(self, min_updates: int = 1) -> Optional[dict]:
        """
        Aggregate pending weight updates using FedAvg.

        Args:
            min_updates: Minimum number of updates required to aggregate

        Returns:
            Aggregated weights, or None if not enough updates
        """
        if len(self.pending_updates) < min_updates:
            return None

        if len(self.pending_updates) == 0:
            return self.global_weights

        # Compute total weight for normalization
        total_weight = sum(w for _, w in self.pending_updates)
        if total_weight <= 0:
            total_weight = len(self.pending_updates)  # Fallback to equal weights

        # Initialize aggregated weights with zeros (same structure as first update)
        first_weights = self.pending_updates[0][0]
        aggregated = {}

        for key in first_weights.keys():
            if isinstance(first_weights[key], torch.Tensor):
                # Initialize with zeros of same shape and dtype
                aggregated[key] = torch.zeros_like(first_weights[key], dtype=torch.float32)
            else:
                # For non-tensor values, just use the first one
                aggregated[key] = first_weights[key]

        # Weighted sum of all updates
        for weights, contribution_weight in self.pending_updates:
            normalized_weight = contribution_weight / total_weight
            for key in weights.keys():
                if isinstance(weights[key], torch.Tensor):
                    aggregated[key] += weights[key].float() * normalized_weight

        # Convert back to original dtypes
        for key in first_weights.keys():
            if isinstance(first_weights[key], torch.Tensor):
                aggregated[key] = aggregated[key].to(first_weights[key].dtype)

        # Update global weights and clear pending updates
        self.global_weights = aggregated
        self.pending_updates = []
        self.aggregation_count += 1

        return aggregated

    def set_global_weights(self, weights: dict) -> None:
        """
        Set the global weights directly.

        Useful for initialization or when resuming from checkpoint.

        Args:
            weights: PyTorch model state dict
        """
        # Handle case where full checkpoint is passed instead of just weights
        if len(weights) < 10:
            if 'model_state_dict' in weights:
                print(f"[Aggregator] WARNING: Received full checkpoint, extracting model_state_dict")
                weights = weights['model_state_dict']
            elif 'model_state' in weights:
                print(f"[Aggregator] WARNING: Received full checkpoint, extracting model_state")
                weights = weights['model_state']

        tensor_count = sum(1 for v in weights.values() if isinstance(v, torch.Tensor))
        print(f"[Aggregator] set_global_weights: {tensor_count} tensors, {len(weights)} total keys")
        if tensor_count < 10:
            print(f"[Aggregator] WARNING: Keys are: {list(weights.keys())}")
        self.global_weights = {k: v.clone() if isinstance(v, torch.Tensor) else v
                               for k, v in weights.items()}


class MomentumAggregator(FedAvgAggregator):
    """
    FedAvg with server-side momentum for improved stability.

    Applies momentum to the aggregated update, which can help smooth out
    noisy updates and improve convergence.
    """

    def __init__(self, momentum: float = 0.9):
        """
        Initialize the momentum aggregator.

        Args:
            momentum: Momentum coefficient (0-1). Higher = more smoothing.
        """
        super().__init__()
        self.momentum = momentum
        self.velocity: Optional[dict] = None

    def aggregate(self, min_updates: int = 1) -> Optional[dict]:
        """
        Aggregate with momentum.

        Args:
            min_updates: Minimum number of updates required

        Returns:
            Aggregated weights with momentum applied
        """
        # First, do regular FedAvg aggregation
        new_weights = super().aggregate(min_updates)
        if new_weights is None:
            return None

        # If this is first aggregation, just return the weights
        if self.velocity is None or self.global_weights is None:
            self.velocity = {k: torch.zeros_like(v) if isinstance(v, torch.Tensor) else 0
                           for k, v in new_weights.items()}
            return new_weights

        # Apply momentum: velocity = momentum * velocity + (new - old)
        # Then: weights = old + velocity
        for key in new_weights.keys():
            if isinstance(new_weights[key], torch.Tensor):
                delta = new_weights[key] - self.global_weights[key]
                self.velocity[key] = self.momentum * self.velocity[key] + delta
                new_weights[key] = self.global_weights[key] + self.velocity[key]

        self.global_weights = new_weights
        return new_weights


def blend_weights(local_weights: dict, global_weights: dict,
                  blend_ratio: float = 0.5) -> dict:
    """
    Blend local and global weights.

    This allows each user to retain some of their local learning while
    incorporating the global consensus.

    Args:
        local_weights: User's local model weights
        global_weights: Aggregated global weights
        blend_ratio: How much to weight global (0.0 = all local, 1.0 = all global)

    Returns:
        Blended weights
    """
    if blend_ratio <= 0.0:
        return local_weights
    if blend_ratio >= 1.0:
        return global_weights

    blended = {}
    local_ratio = 1.0 - blend_ratio

    for key in local_weights.keys():
        if isinstance(local_weights[key], torch.Tensor):
            blended[key] = (local_ratio * local_weights[key].float() +
                          blend_ratio * global_weights[key].float())
            blended[key] = blended[key].to(local_weights[key].dtype)
        else:
            blended[key] = local_weights[key]

    return blended


def compute_weight_delta(old_weights: dict, new_weights: dict) -> dict:
    """
    Compute the delta between two weight dictionaries.

    Useful for sending only the changes instead of full weights.

    Args:
        old_weights: Previous weights
        new_weights: Current weights

    Returns:
        Delta (new - old) for each parameter
    """
    delta = {}
    for key in new_weights.keys():
        if isinstance(new_weights[key], torch.Tensor):
            delta[key] = new_weights[key] - old_weights[key]
        else:
            delta[key] = new_weights[key]
    return delta


def apply_weight_delta(base_weights: dict, delta: dict) -> dict:
    """
    Apply a delta to base weights.

    Args:
        base_weights: Starting weights
        delta: Changes to apply

    Returns:
        Updated weights (base + delta)
    """
    updated = {}
    for key in base_weights.keys():
        if isinstance(base_weights[key], torch.Tensor):
            updated[key] = base_weights[key] + delta[key]
        else:
            updated[key] = delta.get(key, base_weights[key])
    return updated
