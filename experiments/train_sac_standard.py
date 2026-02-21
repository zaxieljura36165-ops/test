"""
Standard SAC training script (non-hierarchical).

Notes:
1. Stochastic policy (entropy regularization).
2. No context-aware noise controller.
3. Output paths aligned with td3_standard.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pickle
from datetime import datetime

import numpy as np

from configs.system_config import SystemConfig
from configs.sac_config import SACConfig
from src.algorithms.sac_standard import StandardSAC
from src.environments.simulation_env import SimulationEnvironment


def train_sac_standard(num_episodes: int = None, save_dir: str = "results/models/sac_standard"):
    """
    Train standard SAC (non-hierarchical).

    Args:
        num_episodes: training episodes (None -> config value)
        save_dir: model output directory
    """
    system_config = SystemConfig()
    sac_config = SACConfig()

    if num_episodes is None:
        num_episodes = sac_config.TRAINING_CONFIG["num_episodes"]

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("results/logs/sac_standard", exist_ok=True)
    os.makedirs("results/plots/sac_standard", exist_ok=True)

    # Create environment
    from configs.mappo_config import MAPPOConfig

    mappo_config = MAPPOConfig()
    env = SimulationEnvironment(system_config, mappo_config)

    # Create SAC agent
    agent = StandardSAC(sac_config, system_config)

    print("=" * 60)
    print("Standard SAC Training (Non-hierarchical)")
    print("=" * 60)
    print(f"Device: {sac_config.DEVICE}")
    print(f"Vehicles: {system_config.NUM_VEHICLES}")
    print(f"RSUs: {system_config.NUM_RSU}")
    print(f"Episodes: {num_episodes}")
    print(f"Buffer size: {sac_config.BUFFER_CONFIG['buffer_size']}")
    print(f"Batch size: {sac_config.BUFFER_CONFIG['batch_size']}")
    print("=" * 60)

    training_stats = {
        "episode_rewards": [],
        "episode_lengths": [],
        "success_rates": [],
        "noise_phases": [],   # keep field alignment
        "noise_scales": [],   # use as entropy coef
        "actor_losses": [],
        "critic_losses": [],
        "objective_values": [],
    }

    window_size = 50
    reward_window = []
    success_window = []
    start_time = time.time()

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            actions = {}
            all_local_states = []

            for vehicle_id in range(system_config.NUM_VEHICLES):
                local_state = env._get_vehicle_state(vehicle_id, state)
                all_local_states.append(local_state)

                action = agent.select_action(local_state, vehicle_id, deterministic=False)
                actions[vehicle_id] = action

            next_state, shared_rewards, done, info = env.step(actions)

            all_next_local_states = [
                env._get_vehicle_state(vid, next_state) for vid in range(system_config.NUM_VEHICLES)
            ]

            joint_raw_actions = [
                actions[vid].get("raw_action", np.zeros(agent.action_dim))
                for vid in range(system_config.NUM_VEHICLES)
            ]

            for vehicle_id in range(system_config.NUM_VEHICLES):
                local_state = env._get_vehicle_state(vehicle_id, state)
                next_local_state = env._get_vehicle_state(vehicle_id, next_state)

                is_noisy = True
                v_reward = shared_rewards.get(vehicle_id, 0.0)

                action_payload = actions.get(vehicle_id, {}).copy()
                action_payload["agent_id"] = vehicle_id
                action_payload["joint_raw_actions"] = [np.array(r).flatten() for r in joint_raw_actions]
                action_payload["all_local_states"] = [
                    ls.detach().cpu().numpy() if hasattr(ls, "detach") else np.array(ls)
                    for ls in all_local_states
                ]
                action_payload["all_next_local_states"] = [
                    ls.detach().cpu().numpy() if hasattr(ls, "detach") else np.array(ls)
                    for ls in all_next_local_states
                ]

                experience = {
                    "local_state": local_state,
                    "action": action_payload,
                    "reward": v_reward,
                    "next_local_state": next_local_state,
                    "done": done,
                    "global_state": state["global_state"],
                    "next_global_state": next_state["global_state"],
                }
                agent.store_experience(experience, agent_id=vehicle_id, is_noisy=is_noisy)

            global_reward = list(shared_rewards.values())[0] if shared_rewards else 0.0
            episode_reward += global_reward
            episode_length += 1
            state = next_state

            # Update networks every 5 steps (aligned with td3_standard)
            if episode_length % 5 == 0:
                agent.update(episode)

        task_stats = env.task_manager.get_task_statistics()

        episode_stats = env.optimization_problem.get_optimization_summary(env.current_time_slot)
        training_stats["objective_values"].append(episode_stats["objective_value"])

        training_stats["episode_rewards"].append(episode_reward)
        training_stats["episode_lengths"].append(episode_length)
        training_stats["success_rates"].append(task_stats["success_rate"])

        # Use noise fields to track entropy coefficient for plotting
        training_stats["noise_phases"].append(0)
        training_stats["noise_scales"].append(agent.get_alpha_value())

        reward_window.append(episode_reward)
        success_window.append(task_stats["success_rate"])
        if len(reward_window) > window_size:
            reward_window.pop(0)
            success_window.pop(0)

        if episode % 10 == 0:
            avg_reward = np.mean(reward_window)
            avg_success = np.mean(success_window)
            elapsed = time.time() - start_time
            print(
                f"Episode {episode:4d} | "
                f"Reward: {episode_reward:8.2f} (Avg: {avg_reward:8.2f}) | "
                f"Length: {episode_length:3d} | "
                f"Success: {task_stats['success_rate']:.2%} (Avg: {avg_success:.2%}) | "
                f"Alpha: {agent.get_alpha_value():.3f} | "
                f"Time: {elapsed/60:.1f}min"
            )

        if episode > 0 and episode % sac_config.TRAINING_CONFIG["save_frequency"] == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_episode_{episode}.pth")
            agent.save_model(checkpoint_path)

            stats_path = os.path.join("results/logs/sac_standard", "training_stats.pkl")
            with open(stats_path, "wb") as f:
                pickle.dump(training_stats, f)

        if episode > 0 and episode % sac_config.TRAINING_CONFIG["eval_frequency"] == 0:
            eval_reward, eval_success = evaluate(
                env, agent, sac_config, system_config, num_episodes=sac_config.TRAINING_CONFIG["num_eval_episodes"]
            )
            print(f"  Eval: Avg Reward = {eval_reward:.2f}, Avg Success = {eval_success:.2%}")

    final_path = os.path.join(save_dir, f"final_model_episode_{num_episodes}.pth")
    agent.save_model(final_path)

    from src.utils.evaluator import PerformanceEvaluator

    evaluator = PerformanceEvaluator(results_dir="results/sac_standard_eval")
    print("\nGenerating evaluation report...")
    evaluator.evaluate_training_performance(training_stats, save_plots=True)
    evaluator.save_training_data(
        training_stats, filename=f"sac_standard_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    stats_path = os.path.join("results/logs/sac_standard", "training_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(training_stats, f)

    plot_training_curves(training_stats, "results/plots/sac_standard")

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Training complete! Total time: {total_time/60:.1f} min")
    print(f"Final model saved to: {final_path}")
    print("=" * 60)

    return agent, training_stats


def evaluate(env, agent, sac_config, system_config, num_episodes: int = 10):
    """Evaluate current policy."""
    agent.set_eval()

    eval_rewards = []
    eval_success_rates = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            actions = {}
            for vehicle_id in range(system_config.NUM_VEHICLES):
                local_state = env._get_vehicle_state(vehicle_id, state)
                action = agent.select_action(local_state, vehicle_id, deterministic=True)
                actions[vehicle_id] = action

            next_state, shared_rewards, done, info = env.step(actions)
            global_reward = list(shared_rewards.values())[0] if shared_rewards else 0.0
            episode_reward += global_reward
            state = next_state

        task_stats = env.task_manager.get_task_statistics()
        eval_rewards.append(episode_reward)
        eval_success_rates.append(task_stats["success_rate"])

    return np.mean(eval_rewards), np.mean(eval_success_rates)


def plot_training_curves(stats: dict, save_dir: str):
    """Plot training curves."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Reward
        ax = axes[0, 0]
        ax.plot(stats["episode_rewards"], alpha=0.3, color="blue")
        window = 50
        if len(stats["episode_rewards"]) >= window:
            smoothed = np.convolve(
                stats["episode_rewards"], np.ones(window) / window, mode="valid"
            )
            ax.plot(range(window - 1, len(stats["episode_rewards"])), smoothed, color="blue", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Episode Reward")
        ax.grid(True, alpha=0.3)

        # Success rate
        ax = axes[0, 1]
        ax.plot(stats["success_rates"], alpha=0.3, color="green")
        if len(stats["success_rates"]) >= window:
            smoothed = np.convolve(
                stats["success_rates"], np.ones(window) / window, mode="valid"
            )
            ax.plot(range(window - 1, len(stats["success_rates"])), smoothed, color="green", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Success Rate")
        ax.set_title("Task Success Rate")
        ax.grid(True, alpha=0.3)

        # Episode length
        ax = axes[0, 2]
        ax.plot(stats["episode_lengths"], alpha=0.5, color="orange")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Length")
        ax.set_title("Episode Length")
        ax.grid(True, alpha=0.3)

        # Actor loss (optional)
        ax = axes[1, 0]
        if len(stats.get("actor_losses", [])) > 0:
            ax.plot(stats["actor_losses"], alpha=0.7, color="red")
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Loss")
        ax.set_title("Actor Loss")
        ax.grid(True, alpha=0.3)

        # Critic loss (optional)
        ax = axes[1, 1]
        if len(stats.get("critic_losses", [])) > 0:
            ax.plot(stats["critic_losses"], alpha=0.7, color="purple")
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Loss")
        ax.set_title("Critic Loss")
        ax.grid(True, alpha=0.3)

        # Entropy coefficient
        ax = axes[1, 2]
        ax.plot(stats["noise_scales"], color="teal", alpha=0.7)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Alpha")
        ax.set_title("Entropy Coefficient")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "sac_standard_training_curves.png"), dpi=150)
        plt.close()

        print(f"Training curves saved to: {save_dir}/sac_standard_training_curves.png")

    except ImportError:
        print("Warning: matplotlib not installed, skip plotting")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train standard SAC (non-hierarchical)")
    parser.add_argument("--episodes", type=int, default=None, help="Training episodes")
    parser.add_argument("--save_dir", type=str, default="results/models/sac_standard", help="Model save dir")

    args = parser.parse_args()

    train_sac_standard(num_episodes=args.episodes, save_dir=args.save_dir)
