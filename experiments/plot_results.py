"""
结果可视化脚本
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import os

sns.set_style("whitegrid")

# 默认只绘制前750个episode
MAX_EPISODES = 2000


def truncate_data(data: list, max_len: int = MAX_EPISODES) -> list:
    """截断数据到指定长度"""
    return data[:max_len] if len(data) > max_len else data


def plot_td3_training_curves(training_stats: dict, save_dir: str, max_episodes: int = MAX_EPISODES):
    """绘制TD3训练曲线"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 截断数据
    rewards = truncate_data(training_stats['episode_rewards'], max_episodes)
    success_rates = truncate_data(training_stats['success_rates'], max_episodes)
    lengths = truncate_data(training_stats['episode_lengths'], max_episodes)
    actor_losses = truncate_data(training_stats.get('actor_losses', []), max_episodes)
    critic_losses = truncate_data(training_stats.get('critic_losses', []), max_episodes)
    
    # 增大平滑窗口，对于 2000 个 Episode，建议使用 100-200 的窗口
    window = 150 if len(rewards) > 150 else (len(rewards) // 5 if len(rewards) > 5 else 1)
    
    # 1. Reward曲线
    ax = axes[0, 0]
    if len(rewards) > window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(rewards, alpha=0.3, color='#2ecc71', label='Episode Rewards')
        ax.plot(range(window-1, len(rewards)), moving_avg, color='#27ae60', 
                linewidth=2, label=f'Moving Average (window={window})')
    else:
        ax.plot(rewards, color='#2ecc71', label='Episode Rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('TD3 Training Reward Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Success Rate曲线
    ax = axes[0, 1]
    if len(success_rates) > window:
        moving_avg_sr = np.convolve(success_rates, np.ones(window)/window, mode='valid')
        ax.plot(success_rates, alpha=0.3, color='#3498db', label='Episode Success Rate')
        ax.plot(range(window-1, len(success_rates)), moving_avg_sr, color='#2980b9',
                linewidth=2, label=f'Moving Average (window={window})')
    else:
        ax.plot(success_rates, color='#3498db', label='Episode Success Rate')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Task Success Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])  # 恢复完整成功率区间，看到从0爬升的过程
    
    # 3. Episode Length曲线
    ax = axes[0, 2]
    if len(lengths) > window:
        moving_avg_len = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax.plot(lengths, alpha=0.3, color='#9b59b6', label='Episode Length')
        ax.plot(range(window-1, len(lengths)), moving_avg_len, color='#8e44ad',
                linewidth=2, label=f'Moving Average (window={window})')
    else:
        ax.plot(lengths, color='#9b59b6', label='Episode Length')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Actor Loss曲线
    ax = axes[1, 0]
    if len(actor_losses) > 0:
        valid_losses = [l for l in actor_losses if l != 0 and l is not None]
        if valid_losses:
            w = min(window, len(valid_losses))
            moving_avg_actor = np.convolve(valid_losses, np.ones(w)/w, mode='valid')
            ax.plot(valid_losses, alpha=0.3, color='#e74c3c', label='Actor Loss')
            ax.plot(range(len(moving_avg_actor)), moving_avg_actor, color='#c0392b',
                   linewidth=2, label=f'Moving Average')
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Loss')
    ax.set_title('TD3 Actor Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Critic Loss曲线
    ax = axes[1, 1]
    if len(critic_losses) > 0:
        valid_losses = [l for l in critic_losses if l != 0 and l is not None]
        if valid_losses:
            w = min(window, len(valid_losses))
            moving_avg_critic = np.convolve(valid_losses, np.ones(w)/w, mode='valid')
            ax.plot(valid_losses, alpha=0.3, color='#3498db', label='Critic Loss')
            ax.plot(range(len(moving_avg_critic)), moving_avg_critic, color='#2980b9',
                   linewidth=2, label=f'Moving Average')
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Loss')
    ax.set_title('TD3 Critic Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Reward分布（只显示收敛后的数据，后半部分）
    ax = axes[1, 2]
    mid_point = len(rewards) // 2
    converged_rewards = rewards[mid_point:]
    ax.hist(converged_rewards, bins=50, alpha=0.7, color='#2ecc71', edgecolor='black')
    ax.axvline(np.mean(converged_rewards), color='#e74c3c', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(converged_rewards):.1f}')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Reward Distribution (Episode {mid_point}-{len(rewards)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'td3_training_curves.png', dpi=300, bbox_inches='tight')
    print(f"TD3训练曲线已保存到: {save_dir / 'td3_training_curves.png'}")
    plt.close()


def plot_maddpg_training_curves(training_stats: dict, save_dir: str, max_episodes: int = MAX_EPISODES):
    """绘制MADDPG训练曲线"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 截断数据
    rewards = truncate_data(training_stats['episode_rewards'], max_episodes)
    success_rates = truncate_data(training_stats['success_rates'], max_episodes)
    lengths = truncate_data(training_stats['episode_lengths'], max_episodes)
    actor_losses = truncate_data(training_stats.get('actor_losses', []), max_episodes)
    critic_losses = truncate_data(training_stats.get('critic_losses', []), max_episodes)
    
    window = min(50, len(rewards) // 10) if len(rewards) > 10 else 1
    
    # 1. Reward曲线
    ax = axes[0, 0]
    if len(rewards) > window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(rewards, alpha=0.3, color='steelblue', label='Episode Rewards')
        ax.plot(range(window-1, len(rewards)), moving_avg, color='orange', 
                linewidth=2, label=f'Moving Average (window={window})')
    else:
        ax.plot(rewards, color='steelblue', label='Episode Rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('MADDPG Training Reward Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Success Rate曲线
    ax = axes[0, 1]
    if len(success_rates) > window:
        moving_avg_sr = np.convolve(success_rates, np.ones(window)/window, mode='valid')
        ax.plot(success_rates, alpha=0.3, color='green', label='Episode Success Rate')
        ax.plot(range(window-1, len(success_rates)), moving_avg_sr, color='darkgreen',
                linewidth=2, label=f'Moving Average (window={window})')
    else:
        ax.plot(success_rates, color='green', label='Episode Success Rate')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Task Success Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Episode Length曲线
    ax = axes[0, 2]
    if len(lengths) > window:
        moving_avg_len = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax.plot(lengths, alpha=0.3, color='purple', label='Episode Length')
        ax.plot(range(window-1, len(lengths)), moving_avg_len, color='darkviolet',
                linewidth=2, label=f'Moving Average (window={window})')
    else:
        ax.plot(lengths, color='purple', label='Episode Length')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Actor Loss曲线
    ax = axes[1, 0]
    if len(actor_losses) > 0:
        valid_losses = [l for l in actor_losses if l != 0 and l is not None]
        if valid_losses:
            w = min(window, len(valid_losses))
            moving_avg_actor = np.convolve(valid_losses, np.ones(w)/w, mode='valid')
            ax.plot(valid_losses, alpha=0.3, color='red', label='Actor Loss')
            ax.plot(range(len(moving_avg_actor)), moving_avg_actor, color='darkred',
                   linewidth=2, label=f'Moving Average')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Actor Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Critic Loss曲线
    ax = axes[1, 1]
    if len(critic_losses) > 0:
        valid_losses = [l for l in critic_losses if l != 0 and l is not None]
        if valid_losses:
            w = min(window, len(valid_losses))
            moving_avg_critic = np.convolve(valid_losses, np.ones(w)/w, mode='valid')
            ax.plot(valid_losses, alpha=0.3, color='blue', label='Critic Loss')
            ax.plot(range(len(moving_avg_critic)), moving_avg_critic, color='darkblue',
                   linewidth=2, label=f'Moving Average')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Critic Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Reward分布
    ax = axes[1, 2]
    ax.hist(rewards, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(rewards), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(rewards):.1f}')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'maddpg_training_curves.png', dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_dir / 'maddpg_training_curves.png'}")
    plt.close()


def plot_algorithm_comparison(stats_dict: dict, save_dir: str, max_episodes: int = MAX_EPISODES):
    """
    对比多个算法
    
    Args:
        stats_dict: 字典，键为算法名称，值为训练统计数据
                   例如: {'TD3': td3_stats, 'MAPPO': mappo_stats, 'MADDPG': maddpg_stats}
        save_dir: 保存目录
        max_episodes: 最大episode数
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义颜色
    colors = {
        'TD3': '#2ecc71',          # 绿色
        'TD3-Std': '#1abc9c',      # 青绿
        'MAPPO': '#e74c3c',        # 红色
        'MADDPG': '#3498db',       # 蓝色
        'PPO': '#9b59b6',          # 紫色
        'SAC': '#f39c12',          # 橙色
        'SAC-Std': '#f1c40f',      # 黄色
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 增大对比图的平滑窗口
    window = 150 if max_episodes > 500 else 50
    
    def _ma_label(name: str) -> str:
        return name if name.upper().startswith('MA-') else f"MA-{name}"

    label_map = {name: _ma_label(name) for name in stats_dict.keys()}

    # 1. Reward对比
    ax = axes[0, 0]
    for algo_name, stats in stats_dict.items():
        rewards = truncate_data(stats['episode_rewards'], max_episodes)
        if len(rewards) > window:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            color = colors.get(algo_name, 'gray')
            ax.plot(range(window-1, len(rewards)), ma,
                    color=color, linewidth=2, label=label_map[algo_name])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(f'Reward Comparison (MA window={window})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Success Rate对比
    ax = axes[0, 1]
    for algo_name, stats in stats_dict.items():
        sr = truncate_data(stats['success_rates'], max_episodes)
        if len(sr) > window:
            ma_sr = np.convolve(sr, np.ones(window)/window, mode='valid')
            color = colors.get(algo_name, 'gray')
            ax.plot(range(window-1, len(sr)), ma_sr,
                    color=color, linewidth=2, label=label_map[algo_name])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])  # 统一使用完整区间
    
    # 3. Episode Length对比
    ax = axes[1, 0]
    for algo_name, stats in stats_dict.items():
        lengths = truncate_data(stats['episode_lengths'], max_episodes)
        if len(lengths) > window:
            ma_len = np.convolve(lengths, np.ones(window)/window, mode='valid')
            color = colors.get(algo_name, 'gray')
            ax.plot(range(window-1, len(lengths)), ma_len,
                    color=color, linewidth=2, label=label_map[algo_name])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 最终性能对比（柱状图）
    ax = axes[1, 1]
    
    algo_names = list(stats_dict.keys())
    n_algos = len(algo_names)
    
    # 计算最后100个episode的性能
    final_rewards = []
    final_sr = []
    final_len = []
    
    for algo_name in algo_names:
        stats = stats_dict[algo_name]
        rewards = truncate_data(stats['episode_rewards'], max_episodes)
        sr = truncate_data(stats['success_rates'], max_episodes)
        lengths = truncate_data(stats['episode_lengths'], max_episodes)
        
        final_rewards.append(np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards))
        final_sr.append(np.mean(sr[-100:]) if len(sr) >= 100 else np.mean(sr))
        final_len.append(np.mean(lengths[-100:]) if len(lengths) >= 100 else np.mean(lengths))
    
    x = np.arange(3)  # 3个指标
    width = 0.8 / n_algos
    
    for i, algo_name in enumerate(algo_names):
        # 归一化显示
        norm_values = [
            final_rewards[i] / 10000 if final_rewards[i] != 0 else 0,
            final_sr[i],
            final_len[i] / 200
        ]
        color = colors.get(algo_name, 'gray')
        offset = (i - n_algos/2 + 0.5) * width
        ax.bar(x + offset, norm_values, width, label=label_map[algo_name], color=color, alpha=0.8)
    
    ax.set_ylabel('Normalized Value')
    ax.set_title(f'Final Performance (last 100 episodes, max {max_episodes})')
    ax.set_xticks(x)
    ax.set_xticklabels(['Reward\n(/10000)', 'Success\nRate', 'Length\n(/200)'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print(f"对比图已保存到: {save_dir / 'algorithm_comparison.png'}")
    plt.close()
    
    # 打印详细对比
    print("\n" + "="*80)
    print(f"算法性能对比（最后100个episodes，截止Episode {max_episodes}）")
    print("="*80)
    header = f"{'指标':<20}"
    for algo_name in algo_names:
        header += f"{label_map[algo_name]:<20}"
    print(header)
    print("-"*80)
    
    reward_line = f"{'平均奖励':<20}"
    sr_line = f"{'成功率':<20}"
    len_line = f"{'平均长度':<20}"
    
    for i in range(n_algos):
        reward_line += f"{final_rewards[i]:<20.2f}"
        sr_line += f"{final_sr[i]:<20.3f}"
        len_line += f"{final_len[i]:<20.1f}"
    
    print(reward_line)
    print(sr_line)
    print(len_line)
    print("="*80)


def plot_td3_vs_others(td3_stats: dict, other_stats: dict = None, save_dir: str = "results/plots"):
    """
    专门绘制TD3与其他算法的对比图
    
    Args:
        td3_stats: TD3的训练统计
        other_stats: 其他算法的统计字典，例如 {'MAPPO': mappo_stats}
        save_dir: 保存目录
    """
    stats_dict = {'TD3': td3_stats}
    if other_stats:
        stats_dict.update(other_stats)
    
    plot_algorithm_comparison(stats_dict, save_dir)


def load_td3_stats(stats_path: str = "results/logs/td3/training_stats.pkl"):
    """加载TD3训练统计数据"""
    import pickle
    with open(stats_path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(description='绘制训练结果图')
    parser.add_argument('--algo', type=str, default='td3', choices=['td3', 'maddpg', 'mappo', 'all'],
                        help='要绘制的算法')
    parser.add_argument('--max-episodes', type=int, default=2000,
                        help='最大绘制的episode数 (默认: 750)')
    parser.add_argument('--stats-path', type=str, default=None,
                        help='训练统计文件路径')
    parser.add_argument('--save-dir', type=str, default='results/plots',
                        help='保存目录')
    
    args = parser.parse_args()
    
    # 更新全局最大episode数
    MAX_EPISODES = args.max_episodes
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if args.algo == 'td3':
        stats_path = args.stats_path or "results/logs/td3/training_stats.pkl"
        print(f"加载TD3训练数据: {stats_path}")
        
        try:
            with open(stats_path, 'rb') as f:
                td3_stats = pickle.load(f)
            
            print(f"数据加载成功，共 {len(td3_stats.get('episode_rewards', []))} 个episodes")
            print(f"只绘制前 {args.max_episodes} 个episodes")
            
            plot_td3_training_curves(td3_stats, save_dir, max_episodes=args.max_episodes)
            print("TD3训练曲线绘制完成！")
            
        except FileNotFoundError:
            print(f"错误: 找不到文件 {stats_path}")
            print("请先运行训练: python experiments/train_td3.py")
            
    elif args.algo == 'maddpg':
        stats_path = args.stats_path or "results/logs/maddpg/training_stats.pkl"
        print(f"加载MADDPG训练数据: {stats_path}")
        
        try:
            with open(stats_path, 'rb') as f:
                maddpg_stats = pickle.load(f)
            
            plot_maddpg_training_curves(maddpg_stats, save_dir, max_episodes=args.max_episodes)
            print("MADDPG训练曲线绘制完成！")
            
        except FileNotFoundError:
            print(f"错误: 找不到文件 {stats_path}")
            
    elif args.algo == 'all':
        # 加载所有可用的算法数据进行对比
        stats_dict = {}
        
        td3_path = "results/logs/td3/training_stats.pkl"
        if Path(td3_path).exists():
            with open(td3_path, 'rb') as f:
                stats_dict['TD3'] = pickle.load(f)
            print("已加载TD3数据")

        td3_std_path = "results/logs/td3_standard/training_stats.pkl"
        if Path(td3_std_path).exists():
            with open(td3_std_path, 'rb') as f:
                stats_dict['TD3-Std'] = pickle.load(f)
            print("已加载TD3-Std数据")
        
        # 尝试加载 MAPPO 数据 (优先查找 .pkl，如果没有则查找最新的 .csv)
        mappo_path = "results/logs/mappo/training_stats.pkl"
        if Path(mappo_path).exists():
            with open(mappo_path, 'rb') as f:
                stats_dict['MAPPO'] = pickle.load(f)
            print("已加载 MAPPO 数据 (.pkl)")
        else:
            # 查找 results/logs 目录下的 training_data_*.csv 文件
            import glob
            csv_files = glob.glob("results/logs/training_data_*.csv")
            if csv_files:
                # 按修改时间排序，取最新的一个
                latest_csv = max(csv_files, key=os.path.getmtime)
                print(f"检测到 MAPPO CSV 数据: {latest_csv}")
                try:
                    import pandas as pd
                    df = pd.read_csv(latest_csv)
                    # 转换格式以匹配 stats_dict
                    mappo_stats = {
                        'episode_rewards': df['episode_rewards'].dropna().tolist(),
                        'success_rates': df['success_rates'].dropna().tolist(),
                        'episode_lengths': df['episode_lengths'].dropna().tolist()
                    }
                    stats_dict['MAPPO'] = mappo_stats
                    print("已加载 MAPPO 数据 (CSV)")
                except Exception as e:
                    print(f"加载 MAPPO CSV 失败: {e}")
        
        maddpg_path = "results/logs/maddpg/training_stats.pkl"
        if Path(maddpg_path).exists():
            with open(maddpg_path, 'rb') as f:
                stats_dict['MADDPG'] = pickle.load(f)
            print("已加载MADDPG数据")

        sac_std_path = "results/logs/sac_standard/training_stats.pkl"
        if Path(sac_std_path).exists():
            with open(sac_std_path, 'rb') as f:
                stats_dict['SAC-Std'] = pickle.load(f)
            print("已加载SAC-Std数据")
        
        if stats_dict:
            plot_algorithm_comparison(stats_dict, save_dir / 'comparison', max_episodes=args.max_episodes)
            print("算法对比图绘制完成！")
        else:
            print("没有找到任何训练数据，请先运行训练")
    
    print(f"\n图片保存在: {save_dir}")

