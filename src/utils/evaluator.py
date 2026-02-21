"""
性能评估和结果可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import os
from datetime import datetime

class PerformanceEvaluator:
    """性能评估器"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.ensure_directories()
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def ensure_directories(self):
        """确保结果目录存在"""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.results_dir}/logs", exist_ok=True)
        os.makedirs(f"{self.results_dir}/models", exist_ok=True)
    
    def evaluate_training_performance(self, training_stats: Dict[str, List], 
                                    save_plots: bool = True) -> Dict[str, Any]:
        """评估训练性能"""
        evaluation_results = {}
        
        # 1. 奖励曲线分析
        if 'episode_rewards' in training_stats:
            rewards = training_stats['episode_rewards']
            evaluation_results['reward_analysis'] = self._analyze_rewards(rewards)
            
            if save_plots:
                self._plot_reward_curve(rewards)
        
        # 2. 回合长度分析
        if 'episode_lengths' in training_stats:
            lengths = training_stats['episode_lengths']
            evaluation_results['length_analysis'] = self._analyze_lengths(lengths)
            
            if save_plots:
                self._plot_length_curve(lengths)
        
        # 3. 成功率分析
        if 'success_rates' in training_stats:
            success_rates = training_stats['success_rates']
            evaluation_results['success_analysis'] = self._analyze_success_rates(success_rates)
            
            if save_plots:
                self._plot_success_rate_curve(success_rates)
        
        # 4. 目标函数值分析
        if 'objective_values' in training_stats:
            objectives = training_stats['objective_values']
            evaluation_results['objective_analysis'] = self._analyze_objectives(objectives)
            
            if save_plots:
                self._plot_objective_curve(objectives)
        
        # 5. 综合性能评估
        evaluation_results['overall_performance'] = self._evaluate_overall_performance(
            training_stats
        )
        
        return evaluation_results
    
    def _analyze_rewards(self, rewards: List[float]) -> Dict[str, float]:
        """分析奖励数据"""
        rewards = np.array(rewards)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'final_reward': rewards[-1] if len(rewards) > 0 else 0,
            'improvement': rewards[-1] - rewards[0] if len(rewards) > 1 else 0,
            'convergence_episode': self._find_convergence_point(rewards)
        }
    
    def _analyze_lengths(self, lengths: List[int]) -> Dict[str, float]:
        """分析回合长度数据"""
        lengths = np.array(lengths)
        
        return {
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'stability': 1.0 / (1.0 + np.std(lengths) / np.mean(lengths)) if np.mean(lengths) > 0 else 0
        }
    
    def _analyze_success_rates(self, success_rates: List[float]) -> Dict[str, float]:
        """分析成功率数据"""
        success_rates = np.array(success_rates)
        
        return {
            'mean_success_rate': np.mean(success_rates),
            'std_success_rate': np.std(success_rates),
            'min_success_rate': np.min(success_rates),
            'max_success_rate': np.max(success_rates),
            'final_success_rate': success_rates[-1] if len(success_rates) > 0 else 0,
            'improvement': success_rates[-1] - success_rates[0] if len(success_rates) > 1 else 0
        }
    
    def _analyze_objectives(self, objectives: List[float]) -> Dict[str, float]:
        """分析目标函数值数据"""
        objectives = np.array(objectives)
        
        return {
            'mean_objective': np.mean(objectives),
            'std_objective': np.std(objectives),
            'min_objective': np.min(objectives),
            'max_objective': np.max(objectives),
            'final_objective': objectives[-1] if len(objectives) > 0 else 0,
            'improvement': objectives[0] - objectives[-1] if len(objectives) > 1 else 0  # 目标是最小化
        }
    
    def _find_convergence_point(self, values: List[float], window_size: int = 100, 
                               threshold: float = 0.01) -> int:
        """找到收敛点"""
        if len(values) < window_size:
            return len(values)
        
        values = np.array(values)
        
        for i in range(window_size, len(values)):
            window = values[i-window_size:i]
            if np.std(window) < threshold * np.mean(window):
                return i
        
        return len(values)
    
    def _evaluate_overall_performance(self, training_stats: Dict[str, List]) -> Dict[str, Any]:
        """评估整体性能"""
        performance = {
            'training_completed': True,
            'convergence_achieved': False,
            'performance_grade': 'C'
        }
        
        # 检查收敛性
        if 'episode_rewards' in training_stats:
            rewards = training_stats['episode_rewards']
            if len(rewards) > 100:
                recent_rewards = rewards[-100:]
                if np.std(recent_rewards) < 0.1 * np.mean(recent_rewards):
                    performance['convergence_achieved'] = True
        
        # 计算性能等级
        if 'success_rates' in training_stats:
            success_rates = training_stats['success_rates']
            if len(success_rates) > 0:
                final_success_rate = success_rates[-1]
                if final_success_rate > 0.9:
                    performance['performance_grade'] = 'A'
                elif final_success_rate > 0.8:
                    performance['performance_grade'] = 'B'
                elif final_success_rate > 0.7:
                    performance['performance_grade'] = 'C'
                else:
                    performance['performance_grade'] = 'D'
        
        return performance
    
    def _plot_reward_curve(self, rewards: List[float]):
        """绘制奖励曲线"""
        plt.figure(figsize=(12, 6))
        
        # 原始奖励曲线
        plt.subplot(1, 2, 1)
        plt.plot(rewards, alpha=0.7, label='Episode Rewards')
        
        # 移动平均
        window_size = min(100, len(rewards) // 10)
        if window_size > 1:
            moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
            plt.plot(moving_avg, label=f'Moving Average (window={window_size})', linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Reward Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 奖励分布
        plt.subplot(1, 2, 2)
        plt.hist(rewards, bins=50, alpha=0.7, density=True)
        plt.xlabel('Reward')
        plt.ylabel('Density')
        plt.title('Reward Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/reward_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_length_curve(self, lengths: List[int]):
        """绘制回合长度曲线"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(lengths, alpha=0.7, label='Episode Lengths')
        
        # 移动平均
        window_size = min(50, len(lengths) // 10)
        if window_size > 1:
            moving_avg = pd.Series(lengths).rolling(window=window_size).mean()
            plt.plot(moving_avg, label=f'Moving Average (window={window_size})', linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Episode Length')
        plt.title('Episode Length Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/length_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_success_rate_curve(self, success_rates: List[float]):
        """绘制成功率曲线"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(success_rates, alpha=0.7, label='Success Rate')
        
        # 移动平均
        window_size = min(50, len(success_rates) // 10)
        if window_size > 1:
            moving_avg = pd.Series(success_rates).rolling(window=window_size).mean()
            plt.plot(moving_avg, label=f'Moving Average (window={window_size})', linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.title('Success Rate Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/success_rate_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_objective_curve(self, objectives: List[float]):
        """绘制目标函数值曲线"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(objectives, alpha=0.7, label='Objective Value')
        
        # 移动平均
        window_size = min(50, len(objectives) // 10)
        if window_size > 1:
            moving_avg = pd.Series(objectives).rolling(window=window_size).mean()
            plt.plot(moving_avg, label=f'Moving Average (window={window_size})', linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Objective Value')
        plt.title('Objective Function Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/objective_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comparison(self, results_dict: Dict[str, Dict[str, List]], 
                       metrics: List[str] = ['episode_rewards', 'success_rates']):
        """绘制不同方法的比较图"""
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            for method_name, stats in results_dict.items():
                if metric in stats:
                    values = stats[metric]
                    
                    # 移动平均
                    window_size = min(50, len(values) // 10)
                    if window_size > 1:
                        moving_avg = pd.Series(values).rolling(window=window_size).mean()
                        axes[i].plot(moving_avg, label=method_name, linewidth=2)
                    else:
                        axes[i].plot(values, label=method_name, linewidth=2)
            
            axes[i].set_xlabel('Episode')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, evaluation_results: Dict[str, Any], 
                       training_stats: Dict[str, List]) -> str:
        """生成评估报告"""
        report = []
        report.append("# 多车辆任务卸载系统 - 性能评估报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 整体性能
        if 'overall_performance' in evaluation_results:
            overall = evaluation_results['overall_performance']
            report.append("## 整体性能评估")
            report.append(f"- 训练完成: {'是' if overall['training_completed'] else '否'}")
            report.append(f"- 收敛达成: {'是' if overall['convergence_achieved'] else '否'}")
            report.append(f"- 性能等级: {overall['performance_grade']}")
            report.append("")
        
        # 详细分析
        for analysis_type, analysis in evaluation_results.items():
            if analysis_type != 'overall_performance':
                report.append(f"## {analysis_type.replace('_', ' ').title()}")
                for metric, value in analysis.items():
                    if isinstance(value, float):
                        report.append(f"- {metric}: {value:.4f}")
                    else:
                        report.append(f"- {metric}: {value}")
                report.append("")
        
        # 训练统计摘要
        report.append("## 训练统计摘要")
        for stat_name, stat_values in training_stats.items():
            if stat_values:
                report.append(f"- {stat_name}: {len(stat_values)} 个数据点")
                report.append(f"  - 平均值: {np.mean(stat_values):.4f}")
                report.append(f"  - 标准差: {np.std(stat_values):.4f}")
                report.append(f"  - 最小值: {np.min(stat_values):.4f}")
                report.append(f"  - 最大值: {np.max(stat_values):.4f}")
        report.append("")
        
        # 建议
        report.append("## 改进建议")
        if 'overall_performance' in evaluation_results:
            overall = evaluation_results['overall_performance']
            if not overall['convergence_achieved']:
                report.append("- 增加训练轮数或调整学习率以促进收敛")
            if overall['performance_grade'] in ['C', 'D']:
                report.append("- 考虑调整网络结构或超参数以提高性能")
        
        report.append("- 可以尝试不同的奖励函数设计")
        report.append("- 考虑增加经验回放缓冲区大小")
        report.append("- 尝试不同的探索策略")
        
        report_text = "\n".join(report)
        
        # 保存报告
        with open(f"{self.results_dir}/logs/evaluation_report.txt", 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text
    
    def save_training_data(self, training_stats: Dict[str, List], 
                          filename: str = None):
        """保存训练数据"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.csv"
        
        # 转换为DataFrame
        max_length = max(len(values) for values in training_stats.values())
        
        data = {}
        for stat_name, stat_values in training_stats.items():
            padded_values = stat_values + [np.nan] * (max_length - len(stat_values))
            data[stat_name] = padded_values
        
        df = pd.DataFrame(data)
        df.to_csv(f"{self.results_dir}/logs/{filename}", index=False)
        
        print(f"训练数据已保存到: {self.results_dir}/logs/{filename}")
    
    def load_training_data(self, filename: str) -> Dict[str, List]:
        """加载训练数据"""
        df = pd.read_csv(f"{self.results_dir}/logs/{filename}")
        
        training_stats = {}
        for column in df.columns:
            values = df[column].dropna().tolist()
            training_stats[column] = values
        
        return training_stats
