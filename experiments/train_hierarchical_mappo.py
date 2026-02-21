"""
分层MAPPO训练主脚本
基于系统建模.md的完整实现
"""

import os
import sys
import torch
import numpy as np
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.system_config import SystemConfig
from configs.mappo_config import MAPPOConfig
from src.environments.simulation_env import SimulationEnvironment, TrainingManager
from src.algorithms.hierarchical_mappo import HierarchicalMAPPO
from src.utils.evaluator import PerformanceEvaluator

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练分层MAPPO多车辆任务卸载系统')
    parser.add_argument('--episodes', type=int, default=1000, help='训练轮数')
    parser.add_argument('--save_freq', type=int, default=100, help='模型保存频率')
    parser.add_argument('--eval_freq', type=int, default=50, help='评估频率')
    parser.add_argument('--config', type=str, default='default', help='配置名称')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的模型路径')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    
    # 创建配置
    system_config = SystemConfig()
    mappo_config = MAPPOConfig()
    mappo_config.DEVICE = device
    
    # 创建环境
    env = SimulationEnvironment(system_config, mappo_config)
    
    # 创建智能体
    agent = HierarchicalMAPPO(mappo_config, system_config)
    
    # 恢复训练（如果指定）
    if args.resume:
        print(f"从 {args.resume} 恢复训练...")
        agent.load_model(args.resume)
    
    # 创建训练管理器
    trainer = TrainingManager(env, agent)
    
    # 创建评估器
    evaluator = PerformanceEvaluator()
    
    print("开始训练...")
    print(f"训练轮数: {args.episodes}")
    print(f"车辆数量: {system_config.NUM_VEHICLES}")
    print(f"RSU数量: {system_config.NUM_RSU}")
    print(f"时隙数量: {system_config.NUM_TIME_SLOTS}")
    print("-" * 50)
    
    # 训练统计
    training_stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'success_rates': [],
        'objective_values': []
    }
    
    # 开始训练
    start_time = datetime.now()
    
    for episode in range(args.episodes):
        # 运行一个回合
        episode_reward, episode_length = trainer._run_episode()
        
        # 记录统计信息
        training_stats['episode_rewards'].append(episode_reward)
        training_stats['episode_lengths'].append(episode_length)
        
        # 获取回合统计 - 直接从task_manager获取
        task_stats = env.task_manager.get_task_statistics()
        training_stats['success_rates'].append(task_stats['success_rate'])
        
        # 获取优化目标值
        episode_stats = env.optimization_problem.get_optimization_summary(env.current_time_slot)
        training_stats['objective_values'].append(episode_stats['objective_value'])
        
        # 更新智能体
        if episode % mappo_config.TRAINING_CONFIG['update_frequency'] == 0:
            agent.update(episode)
        
        # 打印进度（每个episode都打印）
        avg_reward = np.mean(training_stats['episode_rewards'][-10:])
        avg_success = np.mean(training_stats['success_rates'][-10:])
        current_lr = agent._get_current_learning_rate(episode) if hasattr(agent, '_get_current_learning_rate') else 'N/A'
        print(f"Episode {episode:4d} | "
              f"Reward: {episode_reward:8.2f} | "
              f"Avg Reward: {avg_reward:8.2f} | "
              f"Success Rate: {avg_success:.3f} | "
              f"LR: {current_lr:.2e} | "
              f"Length: {episode_length:3d}")
        
        # 每100个episode打印注意力统计
        if episode % 100 == 0 and hasattr(agent.network.high_level_actor, 'get_attention_summary'):
            attention_summary = agent.network.high_level_actor.get_attention_summary()
            if attention_summary.get('attention_enabled', False):
                print(f"      注意力机制统计: 特征冗余度={attention_summary.get('feature_redundancy_score', 0):.3f}, "
                      f"重要特征数={len(attention_summary.get('top_important_features', []))}")
        
        # 评估性能
        if episode % args.eval_freq == 0 and episode > 0:
            print(f"\n评估 Episode {episode}...")
            eval_results = trainer.evaluate(num_episodes=5)
            print(f"评估结果: {eval_results}")
            print("-" * 50)
        
        # 保存模型
        if episode % args.save_freq == 0 and episode > 0:
            model_path = f"results/models/checkpoint_episode_{episode}.pth"
            agent.save_model(model_path)
            print(f"模型已保存到: {model_path}")
    
    # 训练完成
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    print(f"\n训练完成！")
    print(f"总训练时间: {training_time:.2f} 秒")
    print(f"平均每轮时间: {training_time/args.episodes:.2f} 秒")
    
    # 保存最终模型
    final_model_path = f"results/models/final_model_episode_{args.episodes}.pth"
    agent.save_model(final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    # 性能评估
    print("\n开始性能评估...")
    evaluation_results = evaluator.evaluate_training_performance(training_stats, save_plots=True)
    
    # 生成报告
    report = evaluator.generate_report(evaluation_results, training_stats)
    print("\n" + "="*50)
    print("性能评估报告:")
    print("="*50)
    print(report)
    
    # 保存训练数据
    evaluator.save_training_data(training_stats)
    
    print(f"\n所有结果已保存到 results/ 目录")
    print("训练完成！")

if __name__ == "__main__":
    main()
