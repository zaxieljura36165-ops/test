"""
MADDPG训练脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path

from configs.system_config import SystemConfig
from configs.mappo_config import MAPPOConfig  # 复用状态维度配置
from configs.maddpg_config import MADDPGConfig
from src.environments.simulation_env import SimulationEnvironment
from src.algorithms.maddpg import MADDPG


class MADDPGTrainingManager:
    """MADDPG训练管理器"""
    
    def __init__(self, env: SimulationEnvironment, agent: MADDPG, maddpg_config: MADDPGConfig):
        self.env = env
        self.agent = agent
        self.config = maddpg_config
    
    def train(self, num_episodes: int, save_frequency: int = 100):
        """训练MADDPG智能体"""
        
        # 训练统计
        training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'objective_values': [],
            'actor_losses': [],
            'critic_losses': []
        }
        
        # 开始训练
        start_time = datetime.now()
        
        for episode in range(num_episodes):
            # 运行一个回合
            episode_reward, episode_length, losses = self._run_episode()
            
            # 记录统计信息
            training_stats['episode_rewards'].append(episode_reward)
            training_stats['episode_lengths'].append(episode_length)
            
            # 获取回合统计
            task_stats = self.env.task_manager.get_task_statistics()
            training_stats['success_rates'].append(task_stats['success_rate'])
            
            # 获取优化目标值
            episode_stats = self.env.optimization_problem.get_optimization_summary(self.env.current_time_slot)
            training_stats['objective_values'].append(episode_stats['objective_value'])
            
            # 记录损失
            training_stats['actor_losses'].append(losses['actor_loss'])
            training_stats['critic_losses'].append(losses['critic_loss'])
            
            # 打印进度（每个episode都打印）
            avg_reward = np.mean(training_stats['episode_rewards'][-10:])
            avg_success = np.mean(training_stats['success_rates'][-10:])
            noise_scale = self.agent.noise_scale * (self.agent.noise_decay ** self.agent.step_count)
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Avg Reward: {avg_reward:8.2f} | "
                  f"Success Rate: {avg_success:.3f} | "
                  f"Noise: {noise_scale:.4f} | "
                  f"Length: {episode_length:3d} | "
                  f"A_Loss: {losses['actor_loss']:.4f} | "
                  f"C_Loss: {losses['critic_loss']:.4f}")
            
            # 评估性能
            if episode % self.config.EVAL_CONFIG['eval_frequency'] == 0 and episode > 0:
                print(f"\n评估 Episode {episode}...")
                eval_results = self.evaluate(num_episodes=self.config.EVAL_CONFIG['num_eval_episodes'])
                print(f"评估结果: {eval_results}")
                print("-" * 50)
            
            # 保存模型
            if episode % save_frequency == 0 and episode > 0:
                model_dir = Path(self.config.LOG_CONFIG['model_dir'])
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / f'checkpoint_episode_{episode}.pth'
                self.agent.save_model(str(model_path), episode)
        
        print("训练完成！")
        
        # 保存最终统计
        return training_stats
    
    def _run_episode(self):
        """运行一个回合"""
        state = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        while True:
            # 获取动作
            actions = self._get_actions(state, add_noise=True)
            
            # 执行动作
            next_state, shared_rewards, done, info = self.env.step(actions)
            
            # 全局奖励
            global_reward = list(shared_rewards.values())[0] if shared_rewards else 0.0
            
            # 存储经验
            experience = {
                'states': state['local_states'],
                'global_state': state['global_state'],
                'actions': actions,
                'rewards': shared_rewards,
                'next_states': next_state['local_states'],
                'next_global_state': next_state['global_state'],
                'done': done
            }
            self.agent.store_experience(experience)
            
            episode_reward += global_reward
            episode_length += 1
            
            if done:
                break
            
            state = next_state
        
        # 更新网络（每个episode结束后）
        losses = {'actor_loss': 0.0, 'critic_loss': 0.0}
        if len(self.agent.replay_buffer) >= self.agent.batch_size:
            # 多次更新
            update_every = self.config.TRAINING_CONFIG.get('update_every_steps', 1)
            update_every = max(1, update_every)
            num_updates = max(1, episode_length // update_every)
            for _ in range(num_updates):
                update_losses = self.agent.update(episode_length)
                losses['actor_loss'] += update_losses['actor_loss']
                losses['critic_loss'] += update_losses['critic_loss']
            
            # 平均损失
            losses['actor_loss'] /= num_updates
            losses['critic_loss'] /= num_updates
        
        return episode_reward, episode_length, losses
    
    def _get_actions(self, state: dict, add_noise: bool = True):
        """获取所有车辆的动作"""
        actions = {}
        
        for vehicle_id in range(self.env.system_config.NUM_VEHICLES):
            if vehicle_id in self.env.task_manager.active_tasks:
                # 获取车辆局部状态
                vehicle_local_state = state['local_states'][vehicle_id]
                
                # 智能体选择动作
                action = self.agent.select_action(vehicle_local_state, vehicle_id, add_noise)
                actions[vehicle_id] = action
        
        return actions
    
    def evaluate(self, num_episodes: int = 5):
        """评估当前策略"""
        eval_rewards = []
        eval_lengths = []
        eval_success_rates = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            while True:
                actions = self._get_actions(state, add_noise=False)
                next_state, shared_rewards, done, info = self.env.step(actions)
                
                global_reward = list(shared_rewards.values())[0] if shared_rewards else 0.0
                episode_reward += global_reward
                episode_length += 1
                
                if done:
                    break
                
                state = next_state
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            
            # 成功率
            task_stats = self.env.task_manager.get_task_statistics()
            eval_success_rates.append(task_stats['success_rate'])
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_length': np.mean(eval_lengths),
            'avg_success_rate': np.mean(eval_success_rates)
        }


def main():
    parser = argparse.ArgumentParser(description='Train MADDPG for task offloading')
    parser.add_argument('--episodes', type=int, default=1000, help='训练轮数')
    parser.add_argument('--eval_freq', type=int, default=50, help='评估频率')
    parser.add_argument('--save_freq', type=int, default=100, help='保存频率')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    args = parser.parse_args()
    
    # 配置
    system_config = SystemConfig()
    mappo_config = MAPPOConfig()  # 复用状态维度
    maddpg_config = MADDPGConfig()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建环境
    env = SimulationEnvironment(system_config, mappo_config)
    
    # 创建MADDPG智能体
    agent = MADDPG(
        config=maddpg_config.get_config(),
        num_agents=system_config.NUM_VEHICLES,
        state_dim=mappo_config.HIGH_LEVEL_CONFIG['state_dim'],
        global_state_dim=mappo_config.HIGH_LEVEL_CONFIG['state_dim'] * system_config.NUM_VEHICLES,
        device=args.device
    )
    
    # 创建训练管理器
    trainer = MADDPGTrainingManager(env, agent, maddpg_config)
    
    # 开始训练
    print("开始训练...")
    print(f"训练轮数: {args.episodes}")
    print(f"车辆数量: {system_config.NUM_VEHICLES}")
    print(f"RSU数量: {system_config.NUM_RSU}")
    print(f"缓冲区大小: {maddpg_config.REPLAY_CONFIG['buffer_size']}")
    print("-" * 50)
    
    training_stats = trainer.train(
        num_episodes=args.episodes,
        save_frequency=args.save_freq
    )
    
    # 保存训练统计
    import pickle
    stats_dir = Path(maddpg_config.LOG_CONFIG['log_dir'])
    stats_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_dir / 'training_stats.pkl', 'wb') as f:
        pickle.dump(training_stats, f)
    
    print(f"\n训练统计已保存到: {stats_dir / 'training_stats.pkl'}")
    
    # 绘制训练曲线
    from experiments.plot_results import plot_maddpg_training_curves
    plot_maddpg_training_curves(training_stats, maddpg_config.LOG_CONFIG['plot_dir'])


if __name__ == '__main__':
    main()

