"""
情境感知自适应噪声控制器 - 任务卸载场景版本

核心创新：
1. 全局控制：根据训练阶段（成功率）动态调整基础噪声强度
2. 局部控制：根据当前任务情境（负载、优先级、信道等）微调噪声强度
3. 双缓冲区：分离探索性经验和确定性经验，平衡探索与利用

情境分类（任务卸载场景）：
- high_load: 系统高负载，高噪声探索更优卸载策略
- low_load: 系统低负载，低噪声稳定执行
- urgent_task: 紧急任务，中低噪声确保时效
- normal_task: 普通任务，中等噪声
- congested_network: 网络拥塞，较高噪声探索替代方案
- good_channel: 良好信道，低噪声充分利用
- near_deadline: 接近截止时间，低噪声稳定决策
- high_priority: 高优先级任务，中低噪声
"""

import numpy as np
import torch
from collections import deque
from typing import Dict, Any, List, Tuple, Optional


class TaskOffloadNoiseController:
    """
    任务卸载场景的情境感知噪声控制器
    
    核心思想：
    1. 训练阶段控制：Phase1随机探索 → Phase2增强探索 → Phase3稳定微调
    2. 情境感知：根据任务属性和系统状态动态调整噪声
    3. 噪声只降不升：Phase只能升级不能降级，避免策略退化时噪声暴增
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        # ========== 全局噪声控制参数 ==========
        self.phase = 1  # 当前训练阶段
        
        # 阶段转换阈值
        self.phase1_episodes = 50           # 前50轮完全随机探索
        self.phase2_min_episodes = 50      # Phase2最小持续轮数，防止直接跳到Phase3
        self.phase2_threshold = 0.3         # 成功率达到30%进入阶段2
        self.phase3_threshold = 0.6         # 成功率达到60%进入阶段3
        
        # 全局噪声强度
        self.global_noise_phase1 = 0.8      # 阶段1：高噪声
        self.global_noise_phase2_base = 0.4 # 阶段2：基础噪声
        self.global_noise_phase3_start = 0.15  # 阶段3：起始噪声
        self.global_noise_min = 0.02        # 最小噪声
        self.global_noise_phase3_decay = 0.9998  # 阶段3衰减率
        
        # Phase 3 动态噪声跟踪
        self.phase3_current_noise = self.global_noise_phase3_start
        self.phase3_start_episode = None
        
        # ========== 情境噪声控制参数（任务卸载场景）==========
        self.context_noise_factors = {
            'high_load': 0.8,           # 高负载：较高噪声，探索更优卸载策略
            'low_load': 0.3,            # 低负载：低噪声，稳定执行
            'urgent_task': 0.4,         # 紧急任务：中低噪声，确保时效
            'normal_task': 0.6,         # 普通任务：中等噪声
            'congested_network': 0.7,   # 网络拥塞：较高噪声，探索替代方案
            'good_channel': 0.3,        # 良好信道：低噪声，充分利用
            'near_deadline': 0.3,       # 接近截止时间：低噪声，稳定决策
            'high_priority': 0.4,       # 高优先级任务：中低噪声
            'default': 0.5,             # 默认情境
        }
        
        # 情境判断阈值
        self.high_load_threshold = 0.7      # 队列负载 > 70%
        self.low_load_threshold = 0.3       # 队列负载 < 30%
        self.urgent_priority_threshold = 0.7    # 优先级 > 0.7
        self.near_deadline_ratio = 0.3      # 剩余时间/截止时间 < 30%
        self.good_channel_threshold = 0.6   # 信道质量 > 0.6
        self.congested_threshold = 0.8      # 边缘节点负载 > 80%
        
        # ========== 统计信息 ==========
        self.success_history = deque(maxlen=100)
        self.current_episode = 0
        self.current_success_rate = 0.0
        
        # 情境统计
        self.context_counts = {k: 0 for k in self.context_noise_factors.keys()}
        
        # 应用配置
        if config is not None:
            self._apply_config(config)
    
    def _apply_config(self, config: Dict[str, Any]):
        """从配置字典加载参数"""
        if 'phase1_episodes' in config:
            self.phase1_episodes = config['phase1_episodes']
        if 'phase2_min_episodes' in config:
            self.phase2_min_episodes = config['phase2_min_episodes']
        if 'phase2_threshold' in config:
            self.phase2_threshold = config['phase2_threshold']
        if 'phase3_threshold' in config:
            self.phase3_threshold = config['phase3_threshold']
        if 'phase1_noise' in config:
            self.global_noise_phase1 = config['phase1_noise']
        if 'phase2_base_noise' in config:
            self.global_noise_phase2_base = config['phase2_base_noise']
        if 'phase3_start_noise' in config:
            self.global_noise_phase3_start = config['phase3_start_noise']
            self.phase3_current_noise = config['phase3_start_noise']
        if 'min_noise' in config:
            self.global_noise_min = config['min_noise']
        if 'phase3_decay' in config:
            self.global_noise_phase3_decay = config['phase3_decay']
        if 'context_noise_factors' in config:
            self.context_noise_factors.update(config['context_noise_factors'])
        
        # 情境阈值
        if 'high_load_threshold' in config:
            self.high_load_threshold = config['high_load_threshold']
        if 'low_load_threshold' in config:
            self.low_load_threshold = config['low_load_threshold']
        if 'urgent_priority_threshold' in config:
            self.urgent_priority_threshold = config['urgent_priority_threshold']
        if 'near_deadline_ratio' in config:
            self.near_deadline_ratio = config['near_deadline_ratio']
        if 'good_channel_threshold' in config:
            self.good_channel_threshold = config['good_channel_threshold']
    
    def update_training_stats(self, episode: int, is_success: bool):
        """
        更新训练统计信息
        每个episode结束时调用
        """
        self.current_episode = episode
        self.success_history.append(1 if is_success else 0)
        
        if len(self.success_history) > 0:
            self.current_success_rate = np.mean(self.success_history)
        
        self._update_phase()
    
    def _update_phase(self):
        """根据训练进度更新阶段 - Phase只能升不能降"""
        old_phase = self.phase
        
        # 计算目标Phase
        if self.current_episode < self.phase1_episodes:
            target_phase = 1
        elif self.current_episode < self.phase1_episodes + self.phase2_min_episodes:
            target_phase = 2
        elif self.current_success_rate < self.phase2_threshold:
            target_phase = 2
        elif self.current_success_rate < self.phase3_threshold:
            target_phase = 2
        else:
            target_phase = 3
        
        # Phase只能升不能降
        if target_phase > self.phase:
            self.phase = target_phase
        
        if old_phase != self.phase:
            print(f"🔄 噪声控制阶段切换: Phase {old_phase} → Phase {self.phase} "
                  f"(Episode: {self.current_episode}, Success Rate: {self.current_success_rate:.2%})")
            
            if self.phase == 3 and old_phase != 3:
                self.phase3_start_episode = self.current_episode
                self.phase3_current_noise = self.global_noise_phase3_start
                print(f"📉 Phase 3 噪声将从 {self.global_noise_phase3_start:.3f} 持续衰减到 {self.global_noise_min:.3f}")
    
    def get_global_noise_factor(self) -> float:
        """
        获取全局噪声因子
        返回值范围: [global_noise_min, 1]
        """
        if self.phase == 1:
            return self.global_noise_phase1
        elif self.phase == 2:
            progress = min(1.0, self.current_success_rate / self.phase3_threshold)
            noise = self.global_noise_phase2_base * (1.0 - 0.5 * progress)
            return max(self.global_noise_phase3_start, noise)
        else:
            # Phase 3：持续衰减
            self.phase3_current_noise *= self.global_noise_phase3_decay
            self.phase3_current_noise = max(self.phase3_current_noise, self.global_noise_min)
            return self.phase3_current_noise
    
    def classify_context(self, context_info: Dict[str, Any]) -> str:
        """
        根据任务和系统状态判断当前情境
        
        Args:
            context_info: 情境信息字典，包含：
                - queue_load: 队列负载 [0, 1]
                - task_priority: 任务优先级 [0, 1]
                - deadline_ratio: 剩余时间/截止时间 [0, 1]
                - channel_quality: 信道质量 [0, 1]
                - edge_load: 边缘节点负载 [0, 1]
        
        Returns:
            context: 情境类型字符串
        """
        queue_load = context_info.get('queue_load', 0.5)
        task_priority = context_info.get('task_priority', 0.5)
        deadline_ratio = context_info.get('deadline_ratio', 0.5)
        channel_quality = context_info.get('channel_quality', 0.5)
        edge_load = context_info.get('edge_load', 0.5)
        
        # ========== 情境分类逻辑 ==========
        
        # 1. 接近截止时间（最高优先级）
        if deadline_ratio < self.near_deadline_ratio:
            context = 'near_deadline'
        
        # 2. 高优先级任务
        elif task_priority > self.urgent_priority_threshold:
            context = 'high_priority'
        
        # 3. 紧急任务（优先级高且时间紧）
        elif task_priority > 0.5 and deadline_ratio < 0.5:
            context = 'urgent_task'
        
        # 4. 网络拥塞
        elif edge_load > self.congested_threshold:
            context = 'congested_network'
        
        # 5. 系统高负载
        elif queue_load > self.high_load_threshold:
            context = 'high_load'
        
        # 6. 良好信道
        elif channel_quality > self.good_channel_threshold and edge_load < 0.5:
            context = 'good_channel'
        
        # 7. 系统低负载
        elif queue_load < self.low_load_threshold:
            context = 'low_load'
        
        # 8. 默认：普通任务
        else:
            context = 'normal_task'
        
        # 更新统计
        self.context_counts[context] += 1
        
        return context
    
    def get_noise_scale(self, context_info: Dict[str, Any] = None) -> float:
        """
        获取当前步的噪声强度
        
        Args:
            context_info: 情境信息（可选）
        
        Returns:
            noise_scale: 噪声强度 [0, 1]
        """
        # 阶段1：返回最大噪声
        if self.phase == 1:
            return self.global_noise_phase1
        
        # 获取全局噪声因子
        global_factor = self.get_global_noise_factor()
        
        # 如果没有情境信息，返回全局噪声
        if context_info is None:
            return global_factor
        
        # 分类当前情境
        context = self.classify_context(context_info)
        
        # 获取情境噪声因子
        context_factor = self.context_noise_factors.get(context, 0.5)
        
        # 最终噪声 = 全局因子 × 情境因子
        noise_scale = global_factor * context_factor
        
        # 限制范围：确保永远不低于 0.1，防止探索过早停止
        noise_scale = np.clip(noise_scale, 0.1, 1.0)
        
        return noise_scale
    
    def should_use_random_action(self) -> bool:
        """判断是否应该使用完全随机动作（仅Phase1）"""
        return self.phase == 1
    
    def get_deterministic_ratio(self) -> float:
        """
        获取确定性经验的采样比例
        用于双缓冲区采样
        
        Returns:
            ratio: 确定性经验占比
        """
        det_min = 0.1
        det_max = 0.4
        
        if self.phase == 1:
            return 0.0
        elif self.phase == 2:
            progress = min(1.0, self.current_success_rate / self.phase3_threshold)
            return det_min + (det_max - det_min) * 0.5 * progress
        else:
            return det_max
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self.phase == 3:
            current_noise = self.phase3_current_noise
        else:
            current_noise = self.get_global_noise_factor()
        
        return {
            'phase': self.phase,
            'episode': self.current_episode,
            'success_rate': self.current_success_rate,
            'global_noise': current_noise,
            'min_noise': self.global_noise_min,
            'context_counts': dict(self.context_counts)
        }
    
    def reset_episode(self):
        """每个episode开始时重置（如需要）"""
        pass


class DualReplayBuffer:
    """
    双经验回放缓冲区
    
    核心思想：
    - buffer_noisy: 存储带探索噪声的动作产生的经验
    - buffer_deterministic: 存储确定性动作产生的经验
    
    采样策略：
    - 训练早期：主要从noisy buffer采样，鼓励探索
    - 训练后期：增加deterministic buffer采样，稳定策略
    """
    
    def __init__(self, capacity: int, state_dim: int = None, action_dim: int = None):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 双缓冲区
        self.buffer_noisy = []
        self.buffer_deterministic = []
        self.pos_noisy = 0
        self.pos_deterministic = 0
    
    def push(self, state, action, reward, next_state, done,
             global_state=None, next_global_state=None, is_noisy: bool = True):
        """
        添加经验
        
        Args:
            state: 局部状态
            action: 动作（字典）
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
            global_state: 全局状态（可选，CTDE用）
            next_global_state: 下一全局状态
            is_noisy: 是否为探索性经验
        """
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'global_state': global_state,
            'next_global_state': next_global_state
        }
        
        if is_noisy:
            if len(self.buffer_noisy) < self.capacity:
                self.buffer_noisy.append(experience)
            else:
                self.buffer_noisy[self.pos_noisy] = experience
            self.pos_noisy = (self.pos_noisy + 1) % self.capacity
        else:
            if len(self.buffer_deterministic) < self.capacity:
                self.buffer_deterministic.append(experience)
            else:
                self.buffer_deterministic[self.pos_deterministic] = experience
            self.pos_deterministic = (self.pos_deterministic + 1) % self.capacity
    
    def sample(self, batch_size: int, device: torch.device,
               deterministic_ratio: float = 0.3) -> Dict[str, torch.Tensor]:
        """
        从双缓冲区采样
        
        Args:
            batch_size: 批大小
            device: torch设备
            deterministic_ratio: 确定性经验采样比例
        
        Returns:
            批次数据字典
        """
        # 计算各缓冲区采样数量
        n_deterministic = int(batch_size * deterministic_ratio)
        n_noisy = batch_size - n_deterministic
        
        # 确保有足够样本
        n_deterministic = min(n_deterministic, len(self.buffer_deterministic))
        n_noisy = batch_size - n_deterministic
        n_noisy = min(n_noisy, len(self.buffer_noisy))
        
        if n_noisy + n_deterministic < batch_size:
            n_noisy = min(batch_size, len(self.buffer_noisy))
            n_deterministic = 0
        
        # 如果样本不足，返回None
        if n_noisy + n_deterministic == 0:
            return None
        
        samples = []
        
        # 从noisy buffer采样
        if n_noisy > 0 and len(self.buffer_noisy) > 0:
            indices_noisy = np.random.choice(len(self.buffer_noisy), n_noisy, replace=False)
            for idx in indices_noisy:
                samples.append(self.buffer_noisy[idx])
        
        # 从deterministic buffer采样
        if n_deterministic > 0 and len(self.buffer_deterministic) > 0:
            indices_det = np.random.choice(len(self.buffer_deterministic), n_deterministic, replace=False)
            for idx in indices_det:
                samples.append(self.buffer_deterministic[idx])
        
        # 解包数据
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        global_states = []
        next_global_states = []
        
        for exp in samples:
            # 确保state是numpy数组且有正确形状
            state = exp['state']
            next_state = exp['next_state']
            
            if isinstance(state, np.ndarray) and state.size > 0:
                states.append(state.flatten())
            elif isinstance(state, (list, tuple)) and len(state) > 0:
                states.append(np.array(state).flatten())
            else:
                continue  # 跳过无效样本
            
            if isinstance(next_state, np.ndarray) and next_state.size > 0:
                next_states.append(next_state.flatten())
            elif isinstance(next_state, (list, tuple)) and len(next_state) > 0:
                next_states.append(np.array(next_state).flatten())
            else:
                states.pop()  # 移除刚添加的state
                continue
            
            actions.append(exp['action'])
            rewards.append(exp['reward'])
            dones.append(float(exp['done']))
            
            if exp['global_state'] is not None:
                gs = exp['global_state']
                if isinstance(gs, np.ndarray) and gs.size > 0:
                    global_states.append(gs.flatten())
                elif isinstance(gs, (list, tuple)) and len(gs) > 0:
                    global_states.append(np.array(gs).flatten())
            
            if exp['next_global_state'] is not None:
                ngs = exp['next_global_state']
                if isinstance(ngs, np.ndarray) and ngs.size > 0:
                    next_global_states.append(ngs.flatten())
                elif isinstance(ngs, (list, tuple)) and len(ngs) > 0:
                    next_global_states.append(np.array(ngs).flatten())
        
        # 检查是否有有效样本
        if len(states) == 0:
            return None
        
        # 转换为张量
        batch = {
            'states': torch.FloatTensor(np.stack(states)).to(device),
            'actions': actions,  # 保持为列表（因为是字典）
            'rewards': torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            'next_states': torch.FloatTensor(np.stack(next_states)).to(device),
            'dones': torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device),
        }
        
        if global_states and len(global_states) == len(states):
            batch['global_states'] = torch.FloatTensor(np.stack(global_states)).to(device)
        if next_global_states and len(next_global_states) == len(states):
            batch['next_global_states'] = torch.FloatTensor(np.stack(next_global_states)).to(device)
        
        return batch
    
    def __len__(self):
        return len(self.buffer_noisy) + len(self.buffer_deterministic)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计"""
        return {
            'noisy_size': len(self.buffer_noisy),
            'deterministic_size': len(self.buffer_deterministic),
            'total_size': len(self)
        }


def build_context_info_from_state(state: torch.Tensor, task_info: Dict = None,
                                   queue_manager=None, comm_model=None) -> Dict[str, Any]:
    """
    从状态和任务信息构建情境信息
    
    这是一个辅助函数，用于从环境状态构建噪声控制器需要的情境信息
    
    Args:
        state: 状态张量
        task_info: 任务信息字典
        queue_manager: 队列管理器（可选）
        comm_model: 通信模型（可选）
    
    Returns:
        context_info: 情境信息字典
    """
    context_info = {
        'queue_load': 0.5,
        'task_priority': 0.5,
        'deadline_ratio': 0.5,
        'channel_quality': 0.5,
        'edge_load': 0.5,
    }
    
    # 从任务信息提取
    if task_info is not None:
        context_info['task_priority'] = task_info.get('priority', 0.5)
        
        # 计算截止时间比例
        if 'deadline' in task_info and 'arrival_time' in task_info and 'current_time' in task_info:
            remaining = task_info['deadline'] - (task_info['current_time'] - task_info['arrival_time'])
            context_info['deadline_ratio'] = max(0, remaining / task_info['deadline'])
    
    # 从队列管理器提取
    if queue_manager is not None:
        try:
            status = queue_manager.get_global_queue_status()
            total_load = 0
            count = 0
            for vid, vstatus in status.get('vehicles', {}).items():
                total_load += vstatus.get('hpc_length', 0) + vstatus.get('lpc_length', 0)
                count += 1
            if count > 0:
                context_info['queue_load'] = min(1.0, total_load / (count * 10))  # 假设每个队列容量10
            
            # 边缘节点负载
            edge_load = 0
            edge_count = 0
            for eid, estatus in status.get('edges', {}).items():
                edge_load += estatus.get('enc_length', 0)
                edge_count += 1
            if edge_count > 0:
                context_info['edge_load'] = min(1.0, edge_load / (edge_count * 20))
        except:
            pass
    
    # 从通信模型提取信道质量
    if comm_model is not None:
        try:
            # 这里可以添加信道质量的提取逻辑
            pass
        except:
            pass
    
    return context_info
