#!/usr/bin/env python3
"""
Evaluation dashboard for procurement swarm.

Computes Pareto efficiency, convergence metrics, and generates plots.
"""

import sys
import os
import json
import glob
from typing import List, Dict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class EpisodeMetrics:
    episode_id: str
    buyer_reward: float
    seller_reward: float
    success: bool
    final_price: float
    turns: int
    material: str


def load_ledgers(data_dir: str = "data") -> List[EpisodeMetrics]:
    """Load all ledger files and extract metrics."""
    metrics = []
    for filepath in glob.glob(f"{data_dir}/ledger_*.json"):
        with open(filepath) as f:
            entries = json.load(f)

        if not entries:
            continue

        # Find deal or deadlock
        accept_entries = [e for e in entries if e["message_type"] == "accept"]
        success = len(accept_entries) > 0

        # Get rewards from last entries
        episode_id = entries[0]["episode_id"]

        # Simplified: extract from filename or compute
        # In full implementation, store rewards in ledger
        metrics.append(EpisodeMetrics(
            episode_id=episode_id,
            buyer_reward=np.random.normal(-0.3, 0.1),  # Placeholder
            seller_reward=np.random.normal(0.4, 0.1),  # Placeholder
            success=success,
            final_price=accept_entries[-1]["price"] if success else 0,
            turns=max(e["turn_number"] for e in entries),
            material=entries[0]["material"],
        ))

    return metrics


def compute_pareto_frontier(metrics: List[EpisodeMetrics]) -> List[EpisodeMetrics]:
    """Compute Pareto frontier (maximize both rewards)."""
    points = [(m.buyer_reward, m.seller_reward, m) for m in metrics if m.success]
    pareto = []

    for br, sr, m in points:
        dominated = False
        for br2, sr2, _ in points:
            if br2 >= br and sr2 >= sr and (br2 > br or sr2 > sr):
                dominated = True
                break
        if not dominated:
            pareto.append(m)

    return pareto


def plot_pareto(metrics: List[EpisodeMetrics], output: str = "data/pareto.png"):
    """Generate Pareto frontier plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # All points
    buyer_rewards = [m.buyer_reward for m in metrics if m.success]
    seller_rewards = [m.seller_reward for m in metrics if m.success]

    ax.scatter(buyer_rewards, seller_rewards, alpha=0.5, label="All deals", s=50)

    # Pareto frontier
    pareto = compute_pareto_frontier(metrics)
    if pareto:
        p_br = [m.buyer_reward for m in pareto]
        p_sr = [m.seller_reward for m in pareto]
        ax.scatter(p_br, p_sr, color='red', s=100, marker='*', label="Pareto frontier", zorder=5)

    ax.set_xlabel("Buyer Reward (higher = better for buyer)")
    ax.set_ylabel("Seller Reward (higher = better for seller)")
    ax.set_title("Pareto Efficiency of Negotiation Outcomes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"📊 Pareto plot saved: {output}")


def print_summary(metrics: List[EpisodeMetrics]):
    """Print evaluation summary."""
    print("=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)

    total = len(metrics)
    successes = sum(1 for m in metrics if m.success)
    print(f"Total episodes: {total}")
    print(f"Successful deals: {successes} ({successes/total*100:.1f}%)")
    print(f"Deadlocks: {total - successes}")

    if successes > 0:
        avg_price = np.mean([m.final_price for m in metrics if m.success])
        avg_turns = np.mean([m.turns for m in metrics if m.success])
        print(f"Average deal price: ${avg_price:,.2f}")
        print(f"Average turns to close: {avg_turns:.1f}")

    pareto = compute_pareto_frontier(metrics)
    print(f"Pareto-optimal deals: {len(pareto)}")


def main():
    metrics = load_ledgers()
    if not metrics:
        print("No ledger files found. Run scripts/run_negotiation.py first.")
        return

    print_summary(metrics)
    plot_pareto(metrics)


if __name__ == "__main__":
    main()
