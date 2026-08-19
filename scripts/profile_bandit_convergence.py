#!/usr/bin/env python3
"""LinUCB Bandit Convergence Profiling Script.

Runs a fast, in-memory Monte Carlo load test of the LinUCBBandit implementation
across 1,000 simulated procurement rounds without external LLMs or Ray actors.
"""

import sys
import time
import random
from pathlib import Path

# Ensure project root is in sys.path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from mesh.neuro.bandits import LinUCBBandit, BanditContext, NegotiationStrategy


def synthetic_market_oracle(context: BanditContext, action: NegotiationStrategy) -> float:
    """Synthetic market oracle returning reward based on context-strategy alignment.
    
    Returns base reward in [0.0, 1.0] with Gaussian noise (sigma=0.05), clamped to [0.0, 1.0].
    """
    # Base rewards based on context-strategy alignment rules
    base_reward = 0.5  # neutral baseline
    
    # AGGRESSIVE_ANCHOR: best under high urgency, low budget margin
    if action == NegotiationStrategy.AGGRESSIVE_ANCHOR:
        base_reward = 0.7 + 0.2 * context.urgency - 0.15 * context.budget_margin
    # BALANCED_CONCESSION: consistent moderate performance
    elif action == NegotiationStrategy.BALANCED_CONCESSION:
        base_reward = 0.55 + 0.1 * (1 - context.urgency) + 0.05 * context.budget_margin
    # PAYMENT_TERMS_TRADE_OFF: best when budget margin is high (can trade price for terms)
    elif action == NegotiationStrategy.PAYMENT_TERMS_TRADE_OFF:
        base_reward = 0.5 + 0.2 * context.budget_margin - 0.1 * context.urgency
    # RISK_AWARE_PACING: best under high material complexity, low supplier rating
    elif action == NegotiationStrategy.RISK_AWARE_PACING:
        base_reward = 0.6 + 0.15 * context.material_complexity - 0.1 * context.supplier_rating
    # RELATIONSHIP_BUILDING: best under high supplier rating, low urgency
    elif action == NegotiationStrategy.RELATIONSHIP_BUILDING:
        base_reward = 0.55 + 0.2 * context.supplier_rating + 0.1 * (1 - context.urgency)
    
    # Add Gaussian noise and clamp
    noise = np.random.normal(0.0, 0.05)
    reward = base_reward + noise
    return float(np.clip(reward, 0.0, 1.0))


def sample_random_context() -> BanditContext:
    """Sample a random BanditContext covering typical procurement scenarios."""
    return BanditContext(
        urgency=random.uniform(0.0, 1.0),
        budget_margin=random.uniform(0.0, 1.0),
        supplier_rating=random.uniform(0.0, 1.0),
        material_complexity=random.uniform(0.0, 1.0),
        historical_win_rate=random.uniform(0.0, 1.0),
        round_number=random.uniform(0.0, 1.0),
    )


def run_convergence_profile(num_rounds: int = 1000) -> dict:
    """Run Monte Carlo convergence profile for LinUCBBandit.
    
    Returns a dictionary containing all metrics for reporting.
    """
    bandit = LinUCBBandit(alpha=1.0, lambda_reg=1.0)
    
    # Storage for metrics
    round_metrics = []  # list of (round, strategy, reward, select_latency_ms, update_latency_ms)
    window_size = 100
    num_windows = num_rounds // window_size
    
    print(f"Running {num_rounds} Monte Carlo rounds...\n")
    
    for t in range(1, num_rounds + 1):
        context = sample_random_context()
        
        # Measure select_action latency
        start_select = time.perf_counter()
        strategy = bandit.select_action(context)
        select_latency_ms = (time.perf_counter() - start_select) * 1000
        
        # Compute reward from oracle
        reward = synthetic_market_oracle(context, strategy)
        
        # Measure update latency
        start_update = time.perf_counter()
        bandit.update(strategy, context, reward)
        update_latency_ms = (time.perf_counter() - start_update) * 1000
        
        round_metrics.append((t, strategy, reward, select_latency_ms, update_latency_ms))
        
        # Progress indicator
        if t % 200 == 0:
            print(f"  Completed {t}/{num_rounds} rounds...")
    
    print("  Done.\n")
    
    # Compute window-based strategy distribution
    strategy_distribution = []
    for w in range(num_windows):
        start = w * window_size
        end = start + window_size
        window_data = round_metrics[start:end]
        counts = {s: 0 for s in NegotiationStrategy}
        for _, strategy, _, _, _ in window_data:
            counts[strategy] += 1
        percentages = {s.value: (counts[s] / window_size) * 100 for s in NegotiationStrategy}
        strategy_distribution.append(percentages)
    
    # Compute cumulative regret & average reward per window
    window_rewards = []
    for w in range(num_windows):
        start = w * window_size
        end = start + window_size
        window_data = round_metrics[start:end]
        avg_reward = sum(r for _, _, r, _, _ in window_data) / window_size
        window_rewards.append(avg_reward)
    
    # Latency profile
    select_latencies = [m[3] for m in round_metrics]
    update_latencies = [m[4] for m in round_metrics]
    
    latency_profile = {
        "select_p50": float(np.percentile(select_latencies, 50)),
        "select_p95": float(np.percentile(select_latencies, 95)),
        "select_p99": float(np.percentile(select_latencies, 99)),
        "update_p50": float(np.percentile(update_latencies, 50)),
        "update_p95": float(np.percentile(update_latencies, 95)),
        "update_p99": float(np.percentile(update_latencies, 99)),
    }
    
    # Action stats from bandit
    action_stats = bandit.get_action_stats()
    
    return {
        "num_rounds": num_rounds,
        "strategy_distribution": strategy_distribution,
        "window_rewards": window_rewards,
        "latency_profile": latency_profile,
        "action_stats": action_stats,
        "bandit": bandit,
    }


def print_convergence_report(results: dict) -> None:
    """Print structured terminal summary report."""
    num_rounds = results["num_rounds"]
    strategy_dist = results["strategy_distribution"]
    window_rewards = results["window_rewards"]
    latency = results["latency_profile"]
    action_stats = results["action_stats"]
    
    print("=" * 70)
    print("LinUCB BANDIT CONVERGENCE PROFILE REPORT")
    print("=" * 70)
    print(f"\nTotal Rounds: {num_rounds}")
    print(f"Windows: {len(strategy_dist)} x {num_rounds // len(strategy_dist)} rounds\n")
    
    # Strategy Selection Distribution
    print("-" * 70)
    print("STRATEGY SELECTION DISTRIBUTION (per 100-round window)")
    print("-" * 70)
    print(f"{'Window':<8} | {'AGGRESSIVE_ANCHOR':>18} | {'BALANCED_CONCESSION':>18} | "
          f"{'PAYMENT_TERMS':>14} | {'RISK_AWARE':>12} | {'RELATIONSHIP':>12}")
    print(f"{'':<8} | {'':>18} | {'':>18} | {'TRADE_OFF':>14} | {'PACING':>12} | {'BUILDING':>12}")
    print("-" * 70)
    for i, dist in enumerate(strategy_dist):
        print(f"{i+1:<8} | {dist['aggressive_anchor']:>17.1f}% | "
              f"{dist['balanced_concession']:>17.1f}% | "
              f"{dist['payment_terms_trade_off']:>13.1f}% | "
              f"{dist['risk_aware_pacing']:>11.1f}% | "
              f"{dist['relationship_building']:>11.1f}%")
    print()
    
    # Cumulative Regret & Average Reward
    print("-" * 70)
    print("AVERAGE REWARD PER 100-ROUND WINDOW (convergence toward oracle ceiling)")
    print("-" * 70)
    print(f"{'Window':<8} | {'Avg Reward':>12} | {'Cumulative Avg':>15}")
    print("-" * 70)
    cum_sum = 0.0
    for i, avg in enumerate(window_rewards):
        cum_sum += avg
        cum_avg = cum_sum / (i + 1)
        print(f"{i+1:<8} | {avg:>11.4f} | {cum_avg:>14.4f}")
    print()
    
    # Latency Profile
    print("-" * 70)
    print("LATENCY PROFILE (per round, milliseconds)")
    print("-" * 70)
    print(f"{'Operation':<15} | {'P50':>10} | {'P95':>10} | {'P99':>10}")
    print("-" * 70)
    print(f"{'select_action':<15} | {latency['select_p50']:>9.4f} | {latency['select_p95']:>9.4f} | {latency['select_p99']:>9.4f}")
    print(f"{'update':<15} | {latency['update_p50']:>9.4f} | {latency['update_p95']:>9.4f} | {latency['update_p99']:>9.4f}")
    print()
    
    # Action Stats
    print("-" * 70)
    print("FINAL ACTION STATISTICS")
    print("-" * 70)
    print(f"{'Strategy':<28} | {'Count':>6} | {'Total Reward':>12} | {'Avg Reward':>10} | {'Theta Norm':>10}")
    print("-" * 70)
    for strategy, stats in action_stats.items():
        print(f"{strategy:<28} | {stats['count']:>6} | {stats['total_reward']:>12.4f} | "
              f"{stats['avg_reward']:>9.4f} | {stats['theta_norm']:>9.4f}")
    print()
    
    # State Persistence Check
    print("-" * 70)
    print("STATE PERSISTENCE CHECK")
    print("-" * 70)
    test_path = "/tmp/bandit_state_test.json"
    bandit = results["bandit"]
    
    try:
        bandit.save_state(test_path)
        print(f"  save_state() -> OK (written to {test_path})")
        
        loaded_bandit = LinUCBBandit.load_state(test_path)
        print(f"  load_state() -> OK")
        
        # Verify matrix integrity
        original_stats = bandit.get_action_stats()
        loaded_stats = loaded_bandit.get_action_stats()
        
        matrices_match = True
        for action in bandit.strategies:
            a_key = action.value
            if not np.allclose(bandit.A[action], loaded_bandit.A[action]):
                matrices_match = False
                break
            if not np.allclose(bandit.b[action], loaded_bandit.b[action]):
                matrices_match = False
                break
            if not np.allclose(bandit.theta[action], loaded_bandit.theta[action]):
                matrices_match = False
                break
        
        if matrices_match:
            print("  Matrix integrity (A, b, theta) -> VERIFIED")
        else:
            print("  Matrix integrity -> MISMATCH DETECTED")
        
        # Verify action counts and rewards
        counts_match = all(
            original_stats[a]["count"] == loaded_stats[a]["count"]
            for a in original_stats
        )
        rewards_match = all(
            abs(original_stats[a]["total_reward"] - loaded_stats[a]["total_reward"]) < 1e-6
            for a in original_stats
        )
        
        if counts_match and rewards_match:
            print("  Statistics (counts, rewards) -> VERIFIED")
        else:
            print("  Statistics -> MISMATCH DETECTED")
        
        # Cleanup
        Path(test_path).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"  State persistence check FAILED: {e}")
    
    print("\n" + "=" * 70)
    print("PROFILE COMPLETE")
    print("=" * 70)


def main():
    """Main entry point."""
    results = run_convergence_profile(num_rounds=1000)
    print_convergence_report(results)


if __name__ == "__main__":
    main()