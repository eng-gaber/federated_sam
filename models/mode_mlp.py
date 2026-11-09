# -*- coding: utf-8 -*-
"""
MoDE: Mixture of Domain Experts for SAM (Segment Anything Model)
================================================================
Implements token-wise Mixture of Domain Experts on top of the frozen SAM ViT MLP:
    y = y_base + sum_{e in top-k}(w_e * delta_y_e)

Key Architecture Properties:
1. Base MLP weights are frozen and computed ONCE for all tokens.
2. E domain-specific LoRA adapters (default E=5 for Centers C1-C5) on both lin1 and lin2 projections.
3. Lightweight token-wise gating network (D -> hidden_dim -> E).
4. Top-k routing (default k=2) with numerically stable, normalized non-negative weights summing to 1.
5. Mode A: Pure image/token-driven gating (no center ID needed at test time).
6. Mode B: Center-prior gating with lambda weighting for training clients;
   graceful fallback to pure token gating for unseen domains (Center 6).
7. Built-in expert activation frequency logging per client and per block.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMLPLoRA(nn.Module):
    """
    Expert-specific LoRA adapter for SAM's MLPBlock:
    - Projection 1 (lin1: D -> D_mlp) LoRA: A1 (D -> r), B1 (r -> D_mlp)
    - Projection 2 (lin2: D_mlp -> D) LoRA: A2 (D_mlp -> r), B2 (r -> D)
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        mlp_dim: int = 3072,
        r: int = 4,
        lora_alpha: float = 4.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        assert r > 0, "LoRA rank must be positive"
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.r = r
        self.scaling = lora_alpha / float(r)

        # Projection 1 LoRA
        self.lora_a1 = nn.Linear(embedding_dim, r, bias=False)
        self.lora_b1 = nn.Linear(r, mlp_dim, bias=False)

        # Projection 2 LoRA
        self.lora_a2 = nn.Linear(mlp_dim, r, bias=False)
        self.lora_b2 = nn.Linear(r, embedding_dim, bias=False)

        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Kaiming uniform for A, zero for B ensures delta_y_e = 0 at initialization
        nn.init.kaiming_uniform_(self.lora_a1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b1.weight)
        nn.init.kaiming_uniform_(self.lora_a2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b2.weight)

    def forward_expert(
        self,
        x_e: torch.Tensor,
        lin1_base_e: torch.Tensor,
        h_base_e: torch.Tensor,
        base_lin2: nn.Linear,
        act_fn: nn.Module,
    ) -> torch.Tensor:
        """
        Compute expert correction delta_y_e relative to base MLP output:
            delta_y_e = lin2(h_e - h_base_e) + (h_e * A2 * B2) * scaling
        where:
            h_e = act(lin1_base_e + (x_e * A1 * B1) * scaling)
        """
        # Intermediate LoRA delta on lin1
        delta_h_e = self.lora_b1(self.lora_a1(self.dropout(x_e))) * self.scaling
        h_e = act_fn(lin1_base_e + delta_h_e)

        # lin2 delta: base projection of difference (without bias) + expert LoRA on lin2
        delta_lin2 = F.linear(h_e - h_base_e, base_lin2.weight)
        lora_lin2 = self.lora_b2(self.lora_a2(self.dropout(h_e))) * self.scaling

        return delta_lin2 + lora_lin2


class TokenGatingNetwork(nn.Module):
    """
    Token-wise gating network with support for Mode A (token-driven) and
    Mode B (center-prior with graceful unseen-domain fallback).
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        num_experts: int = 5,
        hidden_dim: int = 128,
        routing_mode: str = "mode_a",
        center_prior_lambda: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.routing_mode = routing_mode.lower()
        self.center_prior_lambda = float(center_prior_lambda)

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts, bias=False),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        center_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        tokens: (M, D) flattened token embeddings.
        center_id: Optional integer in [0, num_experts - 1] for training clients in Mode B.
        Returns:
            scores: (M, num_experts) unnormalized routing scores.
        """
        gate_logits = self.net(tokens)  # (M, E)

        if self.routing_mode == "mode_b" and (center_id is not None) and (0 <= center_id < self.num_experts):
            # Mode B: Combine learned gate probabilities with center prior
            p_gate = F.softmax(gate_logits, dim=-1)  # (M, E)

            # One-hot prior for known training center
            p_prior = torch.zeros(self.num_experts, device=tokens.device, dtype=tokens.dtype)
            p_prior[center_id] = 1.0

            lam = self.center_prior_lambda
            p_combined = (1.0 - lam) * p_gate + lam * p_prior.unsqueeze(0)
            scores = torch.log(p_combined + 1e-9)
        else:
            # Mode A or Unseen Center (Center 6) -> Pure token-driven gating
            scores = gate_logits

        return scores


class MoDE_MLP(nn.Module):
    """
    Mixture of Domain Experts MLP Block for SAM ViT-B.
    Replaces the standard MLP in selected transformer blocks.
    """

    def __init__(
        self,
        base_mlp: nn.Module,
        block_idx: int,
        embedding_dim: int = 768,
        mlp_dim: int = 3072,
        num_experts: int = 5,
        r: int = 4,
        lora_alpha: float = 4.0,
        lora_dropout: float = 0.0,
        top_k: int = 2,
        gate_hidden_dim: int = 128,
        routing_mode: str = "mode_a",
        center_prior_lambda: float = 0.2,
    ):
        super().__init__()
        self.block_idx = block_idx
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.num_experts = num_experts
        self.r = r
        self.top_k = min(top_k, num_experts)
        self.routing_mode = routing_mode.lower()
        self.center_prior_lambda = center_prior_lambda
        self.current_center_id: Optional[int] = None

        # 1. Preserve and freeze the pretrained base SAM MLP
        self.base_mlp = base_mlp
        for param in self.base_mlp.parameters():
            param.requires_grad = False

        # 2. Token-wise Gating Network
        self.gating = TokenGatingNetwork(
            embedding_dim=embedding_dim,
            num_experts=num_experts,
            hidden_dim=gate_hidden_dim,
            routing_mode=routing_mode,
            center_prior_lambda=center_prior_lambda,
        )

        # 3. Five Expert-specific LoRA adapters
        self.experts = nn.ModuleList([
            ExpertMLPLoRA(
                embedding_dim=embedding_dim,
                mlp_dim=mlp_dim,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            for _ in range(num_experts)
        ])

        # 4. Expert activation frequency logging
        self.register_buffer("expert_token_counts", torch.zeros(num_experts, dtype=torch.long))
        self.register_buffer("total_routed_tokens", torch.zeros(1, dtype=torch.long))

    def set_center_id(self, center_id: Optional[int]) -> None:
        """Set active client center ID for Mode B routing."""
        self.current_center_id = center_id

    def reset_expert_usage(self) -> None:
        """Reset routing counters for fresh round/epoch logging."""
        self.expert_token_counts.zero_()
        self.total_routed_tokens.zero_()

    def get_expert_usage_distribution(self) -> Dict[int, float]:
        """Return percentage of total token activations assigned to each expert."""
        total = self.total_routed_tokens.item()
        if total == 0:
            return {e: 0.0 for e in range(self.num_experts)}
        counts = self.expert_token_counts.cpu().tolist()
        return {e: float(counts[e]) / float(total) for e in range(self.num_experts)}

    def forward(self, x: torch.Tensor, center_id: Optional[int] = None) -> torch.Tensor:
        """
        x: Input tensor of shape (B, H, W, C) or (B, N, C) matching SAM image encoder.
        Returns:
            Output tensor of identical shape with base MLP + weighted expert corrections.
        """
        orig_shape = x.shape
        D = orig_shape[-1]
        x_flat = x.view(-1, D)  # Flatten to (M, D) where M = B*H*W
        M = x_flat.shape[0]

        # Resolve active center ID (argument overrides attribute)
        active_cid = center_id if center_id is not None else self.current_center_id

        # 1. Base MLP Output (Computed ONCE for all tokens, base weights are frozen)
        lin1_base = self.base_mlp.lin1(x_flat)       # (M, D_mlp)
        h_base = self.base_mlp.act(lin1_base)        # (M, D_mlp)
        y_base = self.base_mlp.lin2(h_base)          # (M, D)

        # 2. Token-wise Routing Scores & Top-k Selection
        scores = self.gating(x_flat, center_id=active_cid)  # (M, E)
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)  # (M, k)
        topk_weights = F.softmax(topk_scores, dim=-1)  # (M, k), non-negative and sums to 1

        # 3. Update Expert Usage Statistics
        if self.training:
            with torch.no_grad():
                for e in range(self.num_experts):
                    hits = (topk_indices == e).sum()
                    self.expert_token_counts[e] += hits
                self.total_routed_tokens += M * self.top_k

        # 4. Compute Weighted Expert Corrections: sum_e(weight_e * delta_y_e)
        delta_y_total = torch.zeros_like(y_base)

        for e in range(self.num_experts):
            # Check which tokens selected expert e in any of their top-k slots
            expert_mask = (topk_indices == e)  # (M, k)
            token_selected = expert_mask.any(dim=-1)  # (M,) boolean

            if not token_selected.any():
                continue

            # Token indices routing to expert e
            token_indices = token_selected.nonzero(as_tuple=False).reshape(-1)
            x_e = x_flat[token_indices]
            lin1_e = lin1_base[token_indices]
            h_base_e = h_base[token_indices]

            # Compute expert correction
            delta_y_e = self.experts[e].forward_expert(
                x_e=x_e,
                lin1_base_e=lin1_e,
                h_base_e=h_base_e,
                base_lin2=self.base_mlp.lin2,
                act_fn=self.base_mlp.act,
            )

            # Weight for expert e on these tokens
            weight_mask = expert_mask[token_indices]  # (N_e, k)
            weight_e = topk_weights[token_indices][weight_mask].unsqueeze(-1)  # (N_e, 1)

            # Accumulate weighted correction: weight_e * delta_y_e
            correction_e = (weight_e * delta_y_e).to(dtype=delta_y_total.dtype)
            delta_y_total.index_add_(0, token_indices, correction_e)

        # 5. Final Output: y = y_base + sum_e(weight_e * delta_y_e)
        y = y_base + delta_y_total

        return y.view(orig_shape)
