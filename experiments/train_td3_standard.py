"""
标准TD3训练脚本（非分层，不使用情境感知噪声与双缓冲区切换）

特点：
1. 仅使用固定高斯探索噪声（标准TD3）
2. 不使用情境噪声控制器
3. 训练/日志/模型输出路径与主算法区分
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
from datetime import datetime
import pickle

from configs.system_config import SystemConfig
from configs.td3_config import TD3Config
from src.algorithms.td3_standard import StandardTD3
from src.environments.simulation_env import SimulationEnvironment


def train_td3_standard(num_episodes: int = None, save_dir: str = 'results/models/td3_standard'):
    """
    训练标准TD3算法（非分层）
    
    Args:
        num_episodes: 训练轮数（None则使用配置值）
        save_dir: 模型保存目录
    """
    # 配置
    system_config = SystemConfig()
    td3_config = TD3Config()
    
    if num_episodes is None:
        num_episodes = td3_config.TRAINING_CONFIG['num_episodes']
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('results/logs/td3_standard', exist_ok=True)
    os.makedirs('results/plots/td3_standard', exist_ok=True)
    
    # 创建环境（复用现有环境，只是算法不同）
    from configs.mappo_config import MAPPOConfig
    mappo_config = MAPPOConfig()
    env = SimulationEnvironment(system_config, mappo_config)
    
    # 创建标准TD3智能体（非分层）
    agent = StandardTD3(td3_config, system_config)
    
    print("=" * 60)
    print("标准TD3训练（非分层） - 固定高斯噪声")
    print("=" * 60)
    print(f"设备: {td3_config.DEVICE}")
    print(f"车辆数量: {system_config.NUM_VEHICLES}")
    print(f"RSU数量: {system_config.NUM_RSU}")
    print(f"训练轮数: {num_episodes}")
    print(f"缓冲区大小: {td3_config.BUFFER_CONFIG['buffer_size']}")
    print(f"批大小: {td3_config.BUFFER_CONFIG['batch_size']}")
    print("=" * 60)
    
    # 探索噪声（线性衰减：0.6 -> 0.1，前60%轮次）
    noise_start = 0.6
    noise_end = 0.1
    decay_episodes = max(1, int(num_episodes * 0.6))
    
    # 训练统计
    training_stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'success_rates': [],
        'noise_phases': [],     # 维持字段一致性
        'noise_scales': [],     # 维持字段一致性
        'actor_losses': [],
        'critic_losses': [],
        'objective_values': [],
    }
    
    # 运行窗口统计
    window_size = 50
    reward_window = []
    success_window = []
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        # 重置环境
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        done = False
        
        # 当前episode噪声
        if episode < decay_episodes:
            progress = episode / decay_episodes
            exploration_noise = noise_start + (noise_end - noise_start) * progress
        else:
            exploration_noise = noise_end

        while not done:
            # 为每个车辆选择动作
            actions = {}
            all_local_states = []
            
            for vehicle_id in range(system_config.NUM_VEHICLES):
                # 获取车辆局部状态
                local_state = env._get_vehicle_state(vehicle_id, state)
                all_local_states.append(local_state)
                
                # 标准TD3：不传入情境信息，使用固定高斯噪声
                action = agent.select_action(
                    local_state, 
                    vehicle_id,
                    deterministic=False,
                    noise_std=exploration_noise
                )
                actions[vehicle_id] = action
            
            # 执行动作
            next_state, shared_rewards, done, info = env.step(actions)

            # 下一状态（所有车辆）
            all_next_local_states = [
                env._get_vehicle_state(vid, next_state) for vid in range(system_config.NUM_VEHICLES)
            ]

            # 联合raw_action
            joint_raw_actions = [
                actions[vid].get('raw_action', np.zeros_like(actions[vid].get('raw_action', np.array([]))))
                for vid in range(system_config.NUM_VEHICLES)
            ]
            
            # 存储经验（每个车辆）
            for vehicle_id in range(system_config.NUM_VEHICLES):
                local_state = env._get_vehicle_state(vehicle_id, state)
                next_local_state = env._get_vehicle_state(vehicle_id, next_state)
                
                # 标准TD3：全部视为探索性经验
                is_noisy = True
                
                v_reward = shared_rewards.get(vehicle_id, 0.0)
                
                # 补充联合信息（集中式Critic）
                action_payload = actions.get(vehicle_id, {}).copy()
                action_payload['agent_id'] = vehicle_id
                action_payload['joint_raw_actions'] = [
                    np.array(r).flatten() for r in joint_raw_actions
                ]
                action_payload['all_local_states'] = [
                    ls.detach().cpu().numpy() if hasattr(ls, 'detach') else np.array(ls)
                    for ls in all_local_states
                ]
                action_payload['all_next_local_states'] = [
                    ls.detach().cpu().numpy() if hasattr(ls, 'detach') else np.array(ls)
                    for ls in all_next_local_states
                ]
                
                experience = {
                    'local_state': local_state,
                    'action': action_payload,
                    'reward': v_reward,
                    'next_local_state': next_local_state,
                    'done': done,
                    'global_state': state['global_state'],
                    'next_global_state': next_state['global_state'],
                }
                agent.store_experience(experience, agent_id=vehicle_id, is_noisy=is_noisy)
            
            # 全局奖励记录用于显示
            global_reward = list(shared_rewards.values())[0] if shared_rewards else 0.0
            episode_reward += global_reward
            episode_length += 1
            state = next_state
            
            # 更新网络：每 5 步更新一次（与主算法对齐）
            if episode_length % 5 == 0:
                agent.update(episode)
        
        # 计算成功率
        task_stats = env.task_manager.get_task_statistics()
        is_success = task_stats['success_rate'] > 0.5
        
        # 获取优化目标值
        episode_stats = env.optimization_problem.get_optimization_summary(env.current_time_slot)
        training_stats['objective_values'].append(episode_stats['objective_value'])
        
        # 记录统计
        training_stats['episode_rewards'].append(episode_reward)
        training_stats['episode_lengths'].append(episode_length)
        training_stats['success_rates'].append(task_stats['success_rate'])
        
        # 噪声统计（固定值）
        training_stats['noise_phases'].append(0)
        training_stats['noise_scales'].append(exploration_noise)
        
        # 更新窗口
        reward_window.append(episode_reward)
        success_window.append(task_stats['success_rate'])
        if len(reward_window) > window_size:
            reward_window.pop(0)
            success_window.pop(0)
        
        # 打印进度
        if episode % 10 == 0:
            avg_reward = np.mean(reward_window)
            avg_success = np.mean(success_window)
            elapsed = time.time() - start_time
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:8.2f} (Avg: {avg_reward:8.2f}) | "
                  f"Length: {episode_length:3d} | "
                  f"Success: {task_stats['success_rate']:.2%} (Avg: {avg_success:.2%}) | "
                  f"Noise: {exploration_noise:.3f} | "
                  f"Time: {elapsed/60:.1f}min")
        
        # 保存检查点
        if episode > 0 and episode % td3_config.TRAINING_CONFIG['save_frequency'] == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_episode_{episode}.pth')
            agent.save_model(checkpoint_path)
            
            # 保存训练统计
            stats_path = os.path.join('results/logs/td3_standard', 'training_stats.pkl')
            with open(stats_path, 'wb') as f:
                pickle.dump(training_stats, f)
        
        # 评估
        if episode > 0 and episode % td3_config.TRAINING_CONFIG['eval_frequency'] == 0:
            eval_reward, eval_success = evaluate(env, agent, td3_config, system_config)
            print(f"  📊 评估结果: Avg Reward = {eval_reward:.2f}, Avg Success = {eval_success:.2%}")
    
    # 保存最终模型
    final_path = os.path.join(save_dir, f'final_model_episode_{num_episodes}.pth')
    agent.save_model(final_path)
    
    # 性能评估（输出到独立目录）
    from src.utils.evaluator import PerformanceEvaluator
    evaluator = PerformanceEvaluator(results_dir='results/td3_standard_eval')
    print("\n开始性能评估报告生成...")
    evaluator.evaluate_training_performance(training_stats, save_plots=True)
    evaluator.save_training_data(training_stats, filename=f"td3_standard_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    # 保存最终统计
    stats_path = os.path.join('results/logs/td3_standard', 'training_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(training_stats, f)
    
    # 绘制训练曲线
    plot_training_curves(training_stats, 'results/plots/td3_standard')
    
    total_time = time.time() - start_time
    print("=" * 60)
    print(f"训练完成! 总时间: {total_time/60:.1f} 分钟")
    print(f"最终模型保存至: {final_path}")
    print("=" * 60)
    
    return agent, training_stats


def evaluate(env, agent, td3_config, system_config, num_episodes: int = 10):
    """评估智能体性能"""
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
                # 评估时使用确定性动作
                action = agent.select_action(
                    local_state, vehicle_id,
                    deterministic=True,
                    noise_std=0.0
                )
                actions[vehicle_id] = action
            
            next_state, shared_rewards, done, info = env.step(actions)
            global_reward = list(shared_rewards.values())[0] if shared_rewards else 0.0
            episode_reward += global_reward
            state = next_state
        
        task_stats = env.task_manager.get_task_statistics()
        eval_rewards.append(episode_reward)
        eval_success_rates.append(task_stats['success_rate'])
    
    return np.mean(eval_rewards), np.mean(eval_success_rates)


def plot_training_curves(stats: dict, save_dir: str):
    """绘制训练曲线"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 奖励曲线
        ax = axes[0, 0]
        ax.plot(stats['episode_rewards'], alpha=0.3, color='blue')
        window = 50
        if len(stats['episode_rewards']) >= window:
            smoothed = np.convolve(stats['episode_rewards'], np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(stats['episode_rewards'])), smoothed, color='blue', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Reward')
        ax.grid(True, alpha=0.3)
        
        # 成功率曲线
        ax = axes[0, 1]
        ax.plot(stats['success_rates'], alpha=0.3, color='green')
        if len(stats['success_rates']) >= window:
            smoothed = np.convolve(stats['success_rates'], np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(stats['success_rates'])), smoothed, color='green', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.set_title('Task Success Rate')
        ax.grid(True, alpha=0.3)
        
        # Episode长度
        ax = axes[0, 2]
        ax.plot(stats['episode_lengths'], alpha=0.5, color='orange')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Length')
        ax.set_title('Episode Length')
        ax.grid(True, alpha=0.3)
        
        # 噪声阶段（固定）
        ax = axes[1, 0]
        ax.plot(stats['noise_phases'], color='red', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Phase')
        ax.set_title('Noise Control Phase')
        ax.grid(True, alpha=0.3)
        
        # 噪声强度
        ax = axes[1, 1]
        ax.plot(stats['noise_scales'], color='purple', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Noise Scale')
        ax.set_title('Exploration Noise')
        ax.grid(True, alpha=0.3)
        
        # 算法信息
        ax = axes[1, 2]
        ax.text(0.5, 0.5, 'Standard TD3\n\nNon-hierarchical',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Algorithm Info')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'td3_standard_training_curves.png'), dpi=150)
        plt.close()
        
        print(f"训练曲线已保存至: {save_dir}/td3_standard_training_curves.png")
        
    except ImportError:
        print("警告: matplotlib未安装，跳过绘图")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练标准TD3算法（非分层）')
    parser.add_argument('--episodes', type=int, default=None, help='训练轮数')
    parser.add_argument('--save_dir', type=str, default='results/models/td3_standard', help='模型保存目录')
    
    args = parser.parse_args()
    
    train_td3_standard(num_episodes=args.episodes, save_dir=args.save_dir)

