"""
分层TD3训练脚本
结合情境感知自适应噪声控制器

用法:
    python experiments/train_td3.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
from datetime import datetime
import pickle
from collections import defaultdict

from configs.system_config import SystemConfig
from configs.td3_config import TD3Config
from src.algorithms.hierarchical_td3 import HierarchicalTD3
from src.algorithms.noise_controller import build_context_info_from_state
from src.environments.simulation_env import SimulationEnvironment


def train_td3(num_episodes: int = None, save_dir: str = 'results/models/td3'):
    """
    训练分层TD3算法
    
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
    os.makedirs('results/logs/td3', exist_ok=True)
    os.makedirs('results/plots/td3', exist_ok=True)
    
    # 创建环境（复用现有环境，只是算法不同）
    from configs.mappo_config import MAPPOConfig
    mappo_config = MAPPOConfig()
    env = SimulationEnvironment(system_config, mappo_config)
    
    # 创建TD3智能体
    agent = HierarchicalTD3(td3_config, system_config)
    
    print("=" * 60)
    print("分层TD3训练 - 情境感知自适应噪声控制")
    print("=" * 60)
    print(f"设备: {td3_config.DEVICE}")
    print(f"车辆数量: {system_config.NUM_VEHICLES}")
    print(f"RSU数量: {system_config.NUM_RSU}")
    print(f"训练轮数: {num_episodes}")
    print(f"缓冲区大小: {td3_config.BUFFER_CONFIG['buffer_size']}")
    print(f"批大小: {td3_config.BUFFER_CONFIG['batch_size']}")
    print("=" * 60)
    
    # 训练统计
    training_stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'success_rates': [],
        'noise_phases': [],
        'noise_scales': [],
        'actor_losses': [],
        'critic_losses': [],
        'objective_values': [],  # 新增：记录优化目标值
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
        
        # === 双缓冲区核心：决定本回合是否使用确定性动作 ===
        # Phase 1: 100% 探索性动作
        # Phase 2: 80% 探索性 + 20% 确定性
        # Phase 3: 60% 探索性 + 40% 确定性
        current_phase = agent.noise_controller.phase
        if current_phase == 1:
            use_deterministic = False
        elif current_phase == 2:
            use_deterministic = np.random.random() < 0.2
        else:  # Phase 3
            use_deterministic = np.random.random() < 0.4
        
        while not done:
            # 为每个车辆选择动作
            actions = {}
            
            for vehicle_id in range(system_config.NUM_VEHICLES):
                # 获取车辆局部状态
                local_state = env._get_vehicle_state(vehicle_id, state)
                
                # 构建情境信息
                context_info = build_context_info_from_state(
                    local_state,
                    task_info=_get_task_info(env, vehicle_id),
                    queue_manager=env.queue_manager,
                    comm_model=env.comm_model
                )
                
                # 选择动作
                action = agent.select_action(
                    local_state, 
                    vehicle_id,
                    context_info=context_info,
                    deterministic=use_deterministic
                )
                actions[vehicle_id] = action
            
            # 执行动作
            next_state, shared_rewards, done, info = env.step(actions)
            
            # 存储经验（每个车辆）
            for vehicle_id in range(system_config.NUM_VEHICLES):
                local_state = env._get_vehicle_state(vehicle_id, state)
                next_local_state = env._get_vehicle_state(vehicle_id, next_state)
                
                # 关键修复：正确标记经验类型
                is_noisy = not use_deterministic
                
                # 关键修复：奖励分配
                # 虽然 shared_rewards 目前是全局奖励，但逻辑上应该按车辆 ID 索引
                # 且为了稳定，可以对奖励进行适当缩放
                v_reward = shared_rewards.get(vehicle_id, 0.0)
                
                experience = {
                    'local_state': local_state,
                    'action': actions.get(vehicle_id, {}),
                    'reward': v_reward,
                    'next_local_state': next_local_state,
                    'done': done,
                    'global_state': state['global_state'],
                    'next_global_state': next_state['global_state'],
                }
                agent.store_experience(experience, is_noisy=is_noisy)
            
            # 全局奖励记录用于显示
            global_reward = list(shared_rewards.values())[0] if shared_rewards else 0.0
            episode_reward += global_reward
            episode_length += 1
            state = next_state
            
            # 更新网络：降低更新频率，每 5 步更新一次
            if episode_length % 5 == 0:
                agent.update(episode)
        
        # 计算成功率
        task_stats = env.task_manager.get_task_statistics()
        is_success = task_stats['success_rate'] > 0.5
        
        # 获取优化目标值
        episode_stats = env.optimization_problem.get_optimization_summary(env.current_time_slot)
        training_stats['objective_values'].append(episode_stats['objective_value'])
        
        # 更新噪声控制器
        agent.update_noise_controller(episode, is_success)
        
        # 记录统计
        training_stats['episode_rewards'].append(episode_reward)
        training_stats['episode_lengths'].append(episode_length)
        training_stats['success_rates'].append(task_stats['success_rate'])
        
        noise_stats = agent.noise_controller.get_stats()
        training_stats['noise_phases'].append(noise_stats['phase'])
        training_stats['noise_scales'].append(noise_stats['global_noise'])
        
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
                  f"Phase: {noise_stats['phase']} | "
                  f"Noise: {noise_stats['global_noise']:.3f} | "
                  f"Time: {elapsed/60:.1f}min")
        
        # 保存检查点
        if episode > 0 and episode % td3_config.TRAINING_CONFIG['save_frequency'] == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_episode_{episode}.pth')
            agent.save_model(checkpoint_path)
            
            # 保存训练统计
            stats_path = os.path.join('results/logs/td3', 'training_stats.pkl')
            with open(stats_path, 'wb') as f:
                pickle.dump(training_stats, f)
        
        # 评估
        if episode > 0 and episode % td3_config.TRAINING_CONFIG['eval_frequency'] == 0:
            eval_reward, eval_success = evaluate(env, agent, td3_config, system_config)
            print(f"  📊 评估结果: Avg Reward = {eval_reward:.2f}, Avg Success = {eval_success:.2%}")
            
            # 记录评估结果到日志（可选，为了对齐 MAPPO 报告）
            with open('results/logs/td3/evaluation_log.txt', 'a') as f:
                f.write(f"Episode {episode}: Reward={eval_reward:.2f}, Success={eval_success:.2%}\n")
    
    # 保存最终模型
    final_path = os.path.join(save_dir, f'final_model_episode_{num_episodes}.pth')
    agent.save_model(final_path)
    
    # 性能评估（新增：调用 evaluator 生成报告，对齐 MAPPO）
    from src.utils.evaluator import PerformanceEvaluator
    evaluator = PerformanceEvaluator(results_dir='results/td3_eval')
    print("\n开始性能评估报告生成...")
    evaluator.evaluate_training_performance(training_stats, save_plots=True)
    evaluator.save_training_data(training_stats, filename=f"td3_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    # 保存最终统计
    stats_path = os.path.join('results/logs/td3', 'training_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(training_stats, f)
    
    # 绘制训练曲线
    plot_training_curves(training_stats, 'results/plots/td3')
    
    total_time = time.time() - start_time
    print("=" * 60)
    print(f"训练完成! 总时间: {total_time/60:.1f} 分钟")
    print(f"最终模型保存至: {final_path}")
    print("=" * 60)
    
    return agent, training_stats


def evaluate(env, agent, td3_config, system_config, num_episodes: int = 10):
    """评估智能体性能"""
    agent.networks.eval()
    
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
                    deterministic=True
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


def _get_task_info(env, vehicle_id: int):
    """从环境获取任务信息"""
    task_info = {
        'priority': 0.5,
        'deadline': 100.0,
        'arrival_time': 0.0,
        'current_time': env.current_time_slot * env.system_config.TIME_SLOT_DURATION,
    }
    
    # 尝试获取活跃任务信息
    if vehicle_id in env.task_manager.active_tasks:
        tasks = env.task_manager.active_tasks[vehicle_id]
        if tasks:
            task = tasks[0]  # 取第一个任务
            task_info['priority'] = task.priority
            task_info['deadline'] = task.deadline
            task_info['arrival_time'] = task.arrival_time
    
    return task_info


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
        
        # 噪声阶段
        ax = axes[1, 0]
        ax.plot(stats['noise_phases'], color='red', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Phase')
        ax.set_title('Noise Control Phase')
        ax.set_yticks([1, 2, 3])
        ax.grid(True, alpha=0.3)
        
        # 噪声强度
        ax = axes[1, 1]
        ax.plot(stats['noise_scales'], color='purple', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Noise Scale')
        ax.set_title('Global Noise Scale')
        ax.grid(True, alpha=0.3)
        
        # 情境统计（如果有）
        ax = axes[1, 2]
        ax.text(0.5, 0.5, 'Context-Aware\nNoise Control\n\nTD3 + Hierarchical\nDecision Making',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Algorithm Info')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'td3_training_curves.png'), dpi=150)
        plt.close()
        
        print(f"训练曲线已保存至: {save_dir}/td3_training_curves.png")
        
    except ImportError:
        print("警告: matplotlib未安装，跳过绘图")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练分层TD3算法')
    parser.add_argument('--episodes', type=int, default=None, help='训练轮数')
    parser.add_argument('--save_dir', type=str, default='results/models/td3', help='模型保存目录')
    
    args = parser.parse_args()
    
    train_td3(num_episodes=args.episodes, save_dir=args.save_dir)

