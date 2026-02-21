"""
算法对比脚本
读取MAPPO的CSV数据和MADDPG的PKL数据，生成奖励对比图
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

sns.set_style("whitegrid")


def load_mappo_csv(csv_path: str):
    """从CSV加载MAPPO训练数据"""
    df = pd.read_csv(csv_path)
    return {
        'episode_rewards': df['episode_rewards'].tolist(),
        'episode_lengths': df['episode_lengths'].tolist(),
        'success_rates': df['success_rates'].tolist(),
    }


def load_maddpg_pkl(pkl_path: str):
    """从PKL加载MADDPG训练数据"""
    with open(pkl_path, 'rb') as f:
        stats = pickle.load(f)
    return stats


def plot_reward_comparison(mappo_stats: dict, maddpg_stats: dict, save_path: str):
    """绘制奖励对比图"""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 提取奖励数据
    mappo_rewards = mappo_stats['episode_rewards']
    maddpg_rewards = maddpg_stats['episode_rewards']
    
    # 计算移动平均
    window = 50
    
    # MAPPO
    if len(mappo_rewards) >= window:
        mappo_ma = np.convolve(mappo_rewards, np.ones(window)/window, mode='valid')
        mappo_episodes = range(window-1, len(mappo_rewards))
    else:
        mappo_ma = mappo_rewards
        mappo_episodes = range(len(mappo_rewards))
    
    # MADDPG
    if len(maddpg_rewards) >= window:
        maddpg_ma = np.convolve(maddpg_rewards, np.ones(window)/window, mode='valid')
        maddpg_episodes = range(window-1, len(maddpg_rewards))
    else:
        maddpg_ma = maddpg_rewards
        maddpg_episodes = range(len(maddpg_rewards))
    
    # 绘制原始数据（透明）
    ax.plot(mappo_rewards, alpha=0.15, color='orange', linewidth=0.5)
    ax.plot(maddpg_rewards, alpha=0.15, color='blue', linewidth=0.5)
    
    # 绘制移动平均
    ax.plot(mappo_episodes, mappo_ma, color='orange', linewidth=2.5, 
            label=f'MAPPO (Moving Avg, window={window})', marker='', markersize=0)
    ax.plot(maddpg_episodes, maddpg_ma, color='blue', linewidth=2.5,
            label=f'MADDPG (Moving Avg, window={window})', marker='', markersize=0)
    
    # 设置标签和标题
    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Reward', fontsize=14)
    ax.set_title('MAPPO vs MADDPG: Training Reward Comparison', fontsize=16, fontweight='bold')
    
    # 添加图例
    ax.legend(fontsize=12, loc='lower right')
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置y轴格式
    ax.ticklabel_format(style='plain', axis='y')
    
    # 添加统计信息文本框
    final_mappo = np.mean(mappo_rewards[-100:])
    final_maddpg = np.mean(maddpg_rewards[-100:])
    
    textstr = f'Last 100 Episodes Avg Reward:\n'
    textstr += f'MAPPO:  {final_mappo:,.0f}\n'
    textstr += f'MADDPG: {final_maddpg:,.0f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # 保存图片
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存到: {save_path}")
    
    plt.show()
    plt.close()
    
    # 打印详细统计
    print("\n" + "="*80)
    print("算法性能对比")
    print("="*80)
    print(f"{'指标':<30} {'MAPPO':<25} {'MADDPG':<25}")
    print("-"*80)
    print(f"{'总Episodes':<30} {len(mappo_rewards):<25} {len(maddpg_rewards):<25}")
    print(f"{'最终奖励 (最后100平均)':<30} {final_mappo:<25.2f} {final_maddpg:<25.2f}")
    print(f"{'最佳奖励':<30} {max(mappo_rewards):<25.2f} {max(maddpg_rewards):<25.2f}")
    print(f"{'最差奖励':<30} {min(mappo_rewards):<25.2f} {min(maddpg_rewards):<25.2f}")
    
    # 成功率（如果有）
    if 'success_rates' in mappo_stats and 'success_rates' in maddpg_stats:
        mappo_sr = np.mean(mappo_stats['success_rates'][-100:])
        maddpg_sr = np.mean(maddpg_stats['success_rates'][-100:])
        print(f"{'成功率 (最后100平均)':<30} {mappo_sr:<25.3f} {maddpg_sr:<25.3f}")
    
    print("="*80)


def main():
    # 文件路径
    mappo_csv = 'results/logs/training_data_20251002_163109.csv'
    maddpg_pkl = 'results/logs/maddpg/training_stats.pkl'
    output_path = 'results/plots/comparison/reward_comparison.png'
    
    print("="*80)
    print("MAPPO vs MADDPG 算法对比")
    print("="*80)
    
    # 加载MAPPO数据
    print(f"\n加载MAPPO数据: {mappo_csv}")
    try:
        mappo_stats = load_mappo_csv(mappo_csv)
        print(f"✓ MAPPO数据加载成功 ({len(mappo_stats['episode_rewards'])} episodes)")
    except Exception as e:
        print(f"✗ 加载MAPPO数据失败: {e}")
        return
    
    # 加载MADDPG数据
    print(f"\n加载MADDPG数据: {maddpg_pkl}")
    try:
        maddpg_stats = load_maddpg_pkl(maddpg_pkl)
        print(f"✓ MADDPG数据加载成功 ({len(maddpg_stats['episode_rewards'])} episodes)")
    except Exception as e:
        print(f"✗ 加载MADDPG数据失败: {e}")
        return
    
    # 生成对比图
    print(f"\n生成奖励对比图...")
    plot_reward_comparison(mappo_stats, maddpg_stats, output_path)
    
    print("\n✓ 对比分析完成！")


if __name__ == '__main__':
    main()

