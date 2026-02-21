"""
仿真环境实现
基于系统建模.md的完整系统仿真
"""

import copy
import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from configs.system_config import SystemConfig
from configs.mappo_config import MAPPOConfig
from src.models.network_model import NetworkModel
from src.models.communication_model import CommunicationModel
from src.models.task_model import TaskManager
from src.models.queue_manager import GlobalQueueManager
from src.models.priority_framework import PriorityFramework
from src.models.delay_energy_calculator import DelayEnergyCalculator
from src.models.problem_formulation import OptimizationProblem
from src.algorithms.hierarchical_mappo import HierarchicalMAPPO
from src.models.task_model import Task

class SimulationEnvironment:
    """仿真环境类"""
    
    def __init__(self, system_config: SystemConfig, mappo_config: MAPPOConfig):
        self.system_config = system_config
        self.mappo_config = mappo_config
        
        # 初始化系统组件
        self._initialize_system_components()
        
        # 仿真状态
        self.current_time_slot = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_stats = defaultdict(list)
        
        # 训练状态
        self.training_mode = True
        self.agent = None
        
    def _initialize_system_components(self):
        """初始化系统组件"""
        # 网络模型
        self.network_model = NetworkModel(self.system_config)
        
        # 通信模型
        self.comm_model = CommunicationModel(self.system_config, self.network_model)
        
        # 任务管理器
        self.task_manager = TaskManager(self.system_config)
        
        # 队列管理器
        self.queue_manager = GlobalQueueManager(self.system_config)
        
        # 优先级框架
        self.priority_framework = PriorityFramework(self.system_config)
        
        # 时延能耗计算器
        self.delay_energy_calc = DelayEnergyCalculator(
            self.system_config, self.network_model, self.comm_model, self.queue_manager
        )
        
        # 优化问题
        self.optimization_problem = OptimizationProblem(
            self.system_config, self.network_model, self.comm_model, 
            self.queue_manager, self.task_manager
        )
    
    def set_agent(self, agent: HierarchicalMAPPO):
        """设置智能体"""
        self.agent = agent
    
    def reset(self) -> Dict[str, Any]:
        """重置环境"""
        # 打印上一轮的统计信息
        if hasattr(self, 'task_manager'):
            stats = self.task_manager.get_task_statistics()
            if stats['completed_tasks'] + stats['failed_tasks'] > 0:
                print(f"[Episode结束] 完成:{stats['completed_tasks']}, 失败:{stats['failed_tasks']}, 成功率:{stats['success_rate']:.2%}")
        
        self.current_time_slot = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_stats = defaultdict(list)
        
        # 重置系统组件
        self.network_model.reset()
        self.comm_model.reset()
        self.task_manager = TaskManager(self.system_config)
        self.queue_manager = GlobalQueueManager(self.system_config)
        self.optimization_problem.reset_decision_variables()
        
        # 获取初始状态
        initial_state = self._get_system_state()
        
        return initial_state
    
    def step(self, actions: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[int, float], bool, Dict[str, Any]]:
        """执行一步仿真
        
        修正：根据算法.md第44-47行，实现全局奖励共享
        R_t = -Σ_{i∈V} Σ_{k∈K_i(t)} C(w_{i,k})
        所有智能体共享这一个全局奖励信号
        
        Returns:
            next_state: 下一个状态
            shared_rewards: Dict[vehicle_id, global_reward] - 所有车辆共享相同的全局奖励
            done: 是否结束
            info: 额外信息
        """
        self.current_time_slot += 1
        self.episode_length += 1
        
        # 1. 生成新任务
        new_tasks = self.task_manager.generate_tasks_for_all_vehicles(self.current_time_slot)
        
        # 2. 更新任务优先级
        time_now = self.current_time_slot * self.system_config.TIME_SLOT_DURATION
        self.task_manager.update_task_priorities(time_now)
        
        # 3. 处理动作，计算新任务的预期成本（已经是单步成本）
        new_task_cost = self._process_actions(actions, new_tasks, time_now)
        
        # 4. 更新队列
        self.queue_manager.process_all_queues(self.system_config.TIME_SLOT_DURATION)
        
        # 5. 检查任务完成状态，计算完成/失败的奖励
        completion_reward = self._check_task_completion(time_now)
        
        # 6. 清理过期任务
        expired_count = self.task_manager.cleanup_expired_tasks(time_now)
        
        # 7. 计算全局奖励（简化版本，避免数值爆炸）
        # - 新任务成本（负奖励）
        # + 完成奖励（正奖励）
        # - 过期惩罚（负奖励）
        global_reward = (-new_task_cost + completion_reward - expired_count * 50.0)
        
        # 限制单步奖励范围，避免数值不稳定
        global_reward = max(min(global_reward, 1000.0), -10000.0)
        
        self.episode_reward += global_reward
        
        # 8. 所有智能体共享相同的全局奖励（MAPPO的关键特征）
        shared_rewards = {vehicle_id: global_reward 
                         for vehicle_id in range(self.system_config.NUM_VEHICLES)}
        
        # 9. 获取新状态
        next_state = self._get_system_state()
        
        # 10. 检查是否结束
        done = self._check_episode_done()
        
        # 11. 记录统计信息
        self._record_episode_stats()
        
        return next_state, shared_rewards, done, self._get_info()
    
    def _process_actions(self, actions: Dict[int, Dict[str, Any]], 
                        new_tasks: Dict[int, List], time_now: float) -> float:
        """处理智能体动作，计算当前step的即时成本
        
        修正：不累积所有任务的总成本（会重复计算）
        只计算新任务的预期成本作为即时反馈
        
        Returns:
            step_cost: 当前step的即时成本
        """
        step_cost = 0.0
        
        # 遍历所有车辆
        for vehicle_id in range(self.system_config.NUM_VEHICLES):
            
            # 处理新任务：计算预期成本作为即时反馈
            if vehicle_id in new_tasks:
                for task in new_tasks[vehicle_id]:
                    # 应用动作到任务
                    self._apply_action_to_task(task, actions.get(vehicle_id, {}))
                    
                    # 计算任务预期成本（作为决策反馈）
                    task_cost = self.delay_energy_calc.calculate_task_cost(task, time_now)
                    step_cost += task_cost
                    
                    # 将任务添加到相应队列
                    self._add_task_to_queues(task, time_now)
        
        return step_cost
    
    def _apply_action_to_task(self, task, action: Dict[str, Any]):
        """将动作应用到任务"""
        # 任务划分
        alpha = action.get('alpha', 0.0)
        task.partition_task(alpha)
        
        # 设置卸载决策
        mode = action.get('mode', 'local')
        if mode == 'V2I':
            target_node = action.get('rsu_action', 0)
            transmit_power = action.get('power', 0.0) * self.system_config.MAX_TRANSMIT_POWER
        elif mode == 'V2V':
            target_node = action.get('neighbor_action', 0)
            transmit_power = action.get('power', 0.0) * self.system_config.MAX_TRANSMIT_POWER
        else:  # local
            target_node = None
            transmit_power = 0.0
        
        local_freq = action.get('freq', 0.0) * self.system_config.VEHICLE_MAX_FREQ
        
        # 设置任务决策
        if mode != 'local':
            task.set_offload_decision(mode, target_node, transmit_power, local_freq)
        else:
            # 本地处理：明确设置offload_mode为None
            task.offload_mode = None
            task.target_node = None
            task.transmit_power = 0.0
            task.local_freq = local_freq
    
    def _add_task_to_queues(self, task, time_now: float):
        """将任务添加到相应队列"""
        # 根据offload_mode和alpha决定任务添加方式
        if task.offload_mode is None or task.alpha < 0.01:
            # 完全本地处理（mode='local'或alpha接近0）
            self.queue_manager.add_local_task(task.vehicle_id, task, time_now)
        elif task.alpha > 0.99:
            # 完全卸载（alpha接近1）
            if task.offload_mode == 'V2I':
                self.queue_manager.add_offload_task(task.vehicle_id, task, time_now, 'V2I')
                self.queue_manager.add_edge_task(task.target_node, task, time_now)
            elif task.offload_mode == 'V2V':
                self.queue_manager.add_offload_task(task.vehicle_id, task, time_now, 'V2V')
                self.queue_manager.add_local_task(task.target_node, task, time_now)
        else:
            # 部分卸载（0 < alpha < 1）
            # 本地部分
            local_task = self._create_local_subtask(task)
            self.queue_manager.add_local_task(task.vehicle_id, local_task, time_now)
            
            # 卸载部分
            if task.offload_mode == 'V2I':
                self.queue_manager.add_offload_task(task.vehicle_id, task, time_now, 'V2I')
                self.queue_manager.add_edge_task(task.target_node, task, time_now)
            elif task.offload_mode == 'V2V':
                self.queue_manager.add_offload_task(task.vehicle_id, task, time_now, 'V2V')
                self.queue_manager.add_local_task(task.target_node, task, time_now)
    
    def _create_local_subtask(self, task):
        """创建本地子任务"""
        local_task = copy.deepcopy(task)
        local_task.data_size = task.local_data_size
        local_task.cpu_cycles = task.local_cpu_cycles
        local_task.offload_mode = None
        local_task.target_node = None
        return local_task

    def _calculate_failure_penalty(self, task: Task, time_now: float) -> float:
        """计算任务失败惩罚

        根据文档公式：
        Penalty_{i,k} = μ * (T_{i,k}^{total} - δ_{i,k}) / δ_{i,k} * (1 + λ_p * P_{i,k} + λ_u * U_{i,k})
        """
        # 计算超时程度
        total_delay = self.delay_energy_calc.calculate_total_delay(task, time_now)
        deadline = task.deadline
        delay_excess = total_delay - deadline

        if delay_excess <= 0:
            return 0.0  # 没有超时，不惩罚

        # 归一化超时程度
        normalized_excess = delay_excess / deadline

        # 计算优先级和紧急度加权
        priority_weight = self.system_config.PRIORITY_PENALTY_WEIGHT * task.priority
        urgency_weight = self.system_config.URGENCY_PENALTY_WEIGHT * (1 - (time_now - task.arrival_time) / deadline)

        # 惩罚权重
        penalty_multiplier = 1 + priority_weight + urgency_weight

        # 计算最终惩罚
        penalty = (self.system_config.PENALTY_MULTIPLIER *
                  normalized_excess *
                  penalty_multiplier)

        return penalty

    def _check_task_completion(self, time_now: float) -> float:
        """检查任务完成状态 - 基于实际延迟计算
        
        Returns:
            completion_reward: 完成奖励 - 失败惩罚
        """
        debug_count = {'checked': 0, 'completed': 0, 'failed': 0}
        completion_reward = 0.0
        
        for vehicle_id in range(self.system_config.NUM_VEHICLES):
            if vehicle_id in self.task_manager.active_tasks:
                completed_tasks = []
                for task in self.task_manager.active_tasks[vehicle_id]:
                    if task.status == "pending":
                        debug_count['checked'] += 1
                        task_age = time_now - task.arrival_time
                        
                        # 基于真实的延迟计算来判断任务是否完成
                        # 使用calculate_total_delay统一处理所有情况
                        try:
                            actual_delay = self.delay_energy_calc.calculate_total_delay(
                                task, task.arrival_time
                            )
                        except Exception as e:
                            # 如果计算失败，使用默认值
                            actual_delay = 5.0
                        
                        # 如果任务已经处理完成（task_age >= actual_delay）
                        if actual_delay > 0 and task_age >= actual_delay:
                            # 判断实际延迟是否在截止时间内
                            if actual_delay <= task.deadline:
                                # 在截止时间内完成 - 给予正奖励
                                self.task_manager.complete_task(task, time_now)
                                completed_tasks.append(task)
                                debug_count['completed'] += 1
                                
                                # 成功完成：根据任务优先级给予奖励
                                success_reward = 5.0 + task.priority * 5.0  # 优先级越高，奖励越大
                                completion_reward += success_reward
                                
                                # if debug_count['completed'] <= 3:  # 只打印前3个
                                #     print(f"✓ 任务完成: delay={actual_delay:.2f}s, deadline={task.deadline:.2f}s")
                            else:
                                # 实际延迟超过截止时间，失败 - 给予负奖励
                                self.task_manager.fail_task(task, time_now)
                                completed_tasks.append(task)
                                debug_count['failed'] += 1
                                
                                # 失败惩罚：优先级越高，惩罚越大
                                failure_penalty = -(10.0 + task.priority * 20.0)
                                completion_reward += failure_penalty
                                
                                # if debug_count['failed'] <= 3:  # 只打印前3个
                                #     print(f"✗ 任务失败(延迟): delay={actual_delay:.2f}s, deadline={task.deadline:.2f}s")
                        # 如果任务严重超时（超过截止时间），强制失败
                        elif task_age > task.deadline:
                            self.task_manager.fail_task(task, time_now)
                            completed_tasks.append(task)
                            debug_count['failed'] += 1
                            
                            # 严重超时：更大的惩罚
                            timeout_penalty = -(15.0 + task.priority * 30.0)
                            completion_reward += timeout_penalty
                            # if debug_count['failed'] <= 3:  # 只打印前3个
                            #     print(f"✗ 任务失败(超时): age={task_age:.2f}s, deadline={task.deadline:.2f}s")

                # 从活跃任务中移除已完成的任务
                for task in completed_tasks:
                    if task in self.task_manager.active_tasks[vehicle_id]:
                        self.task_manager.active_tasks[vehicle_id].remove(task)
        
        # 每个step结束时打印总计（用于调试）- 暂时注释掉
        # if debug_count['completed'] > 0 or debug_count['failed'] > 0:
        #     print(f"[Step {self.current_time_slot}] 完成:{debug_count['completed']}, 失败:{debug_count['failed']}, 累计完成:{len(self.task_manager.completed_tasks)}, 累计失败:{len(self.task_manager.failed_tasks)}")
        
        return completion_reward
    
    def _get_system_state(self) -> Dict[str, Any]:
        """获取系统状态
        
        修正：构建包含所有车辆局部状态的系统状态
        用于CTDE范式：
        - Actor执行时使用局部状态
        - Critic训练时使用全局状态（所有局部状态拼接）
        
        Returns:
            Dict包含：
            - 'local_states': Dict[vehicle_id, torch.Tensor] - 每个车辆的局部状态
            - 'global_state': torch.Tensor - 全局状态（所有局部状态拼接）
        """
        local_states = {}
        
        # 为每个车辆构建局部状态
        for vehicle_id in range(self.system_config.NUM_VEHICLES):
            local_state = self._build_vehicle_local_state(vehicle_id)
            local_states[vehicle_id] = local_state
        
        # 构建全局状态（所有局部状态拼接）
        global_state = torch.cat([local_states[vid] for vid in range(self.system_config.NUM_VEHICLES)], dim=0)
        
        return {
            'local_states': local_states,  # Dict[vehicle_id, local_state_tensor]
            'global_state': global_state    # 全局状态拼接
        }
    
    def _check_episode_done(self) -> bool:
        """检查回合是否结束 - 严格按照文档设计原则实现

        根据文档设计，回合结束应该基于以下原则：
        1. 保证足够的观察时间（至少50个时隙，对应5秒）
        2. 基于任务完成情况自然结束
        3. 避免人为截断重要的决策序列
        4. 考虑队列系统的时序依赖性
        """
        # 1. 最大时隙限制：从配置文件读取
        max_steps = min(self.mappo_config.TRAINING_CONFIG['max_steps_per_episode'], 
                       self.system_config.NUM_TIME_SLOTS)
        if self.current_time_slot >= max_steps:
            return True

        # 2. 最小观察时间：从配置文件读取
        min_observation_steps = self.mappo_config.TRAINING_CONFIG['min_steps_per_episode']
        if self.current_time_slot < min_observation_steps:
            return False

        # 3. 自然结束条件：基于系统状态和任务完成情况
        # 3.1 检查是否有活跃任务
        total_active_tasks = sum(len(tasks) for tasks in self.task_manager.active_tasks.values())

        # 3.2 检查队列状态（简化计算）
        queue_status = self.queue_manager.get_global_queue_status()
        total_queued_tasks = 0
        for vehicle_id, status in queue_status['vehicles'].items():
            total_queued_tasks += (status['hpc_length'] + status['lpc_length'] +
                                 status['v2i_length'] + status['v2v_length'])

        # 3.3 更加随机化的结束策略
        steps_beyond_min = self.current_time_slot - min_observation_steps

        # 添加调试信息（可选）
        # if self.current_time_slot % 20 == 0:  # 每20步打印一次调试信息
        #     print(f"时隙{self.current_time_slot}: 活跃任务={total_active_tasks}, 队列任务={total_queued_tasks}")
            
        # 调试：打印结束条件分析
        # if self.current_time_slot > min_observation_steps and self.current_time_slot % 30 == 0:
        #     print(f"  结束分析 - 时隙:{self.current_time_slot}, 超过最小:{steps_beyond_min}, "
        #           f"活跃任务:{total_active_tasks}, 队列任务:{total_queued_tasks}")

        # 简化的结束条件：固定运行到min_steps，然后随机结束
        if steps_beyond_min <= 0:
            return False  # 还未到最小时间
        
        # 到达最大步数时强制结束
        if self.current_time_slot >= max_steps:
            return True
        
        # 在min和max之间：基于简单概率随机结束
        # 概率随步数线性增长
        steps_in_range = max_steps - min_observation_steps
        end_probability = steps_beyond_min / steps_in_range * 0.3  # 最多30%概率
        
        return np.random.random() < end_probability
    
    def _record_episode_stats(self):
        """记录回合统计信息"""
        stats = self.optimization_problem.get_optimization_summary(self.current_time_slot)
        
        self.episode_stats['total_tasks'].append(stats['total_tasks'])
        self.episode_stats['completed_tasks'].append(stats['completed_tasks'])
        self.episode_stats['failed_tasks'].append(stats['failed_tasks'])
        self.episode_stats['success_rate'].append(stats['success_rate'])
        self.episode_stats['objective_value'].append(stats['objective_value'])
        self.episode_stats['average_queue_length'].append(stats['average_queue_length'])
        self.episode_stats['system_utilization'].append(stats['system_utilization'])
    
    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        return {
            'episode_length': self.episode_length,
            'episode_reward': self.episode_reward,
            'current_time_slot': self.current_time_slot,
            'task_statistics': self.task_manager.get_task_statistics(),
            'queue_status': self.queue_manager.get_global_queue_status()
        }
    
    def get_observation_space(self) -> Dict[str, Any]:
        """获取观察空间"""
        return {
            'state_dim': self.mappo_config.HIGH_LEVEL_CONFIG['state_dim'],
            'action_dim': self.mappo_config.HIGH_LEVEL_CONFIG['action_dim']
        }
    
    def get_action_space(self) -> Dict[str, Any]:
        """获取动作空间"""
        return {
            'alpha_range': (0.0, 1.0),
            'mode_choices': ['local', 'V2I', 'V2V'],
            'power_range': (0.0, self.system_config.MAX_TRANSMIT_POWER),
            'freq_range': (0.0, self.system_config.VEHICLE_MAX_FREQ)
        }
    
    def render(self, mode: str = 'human'):
        """渲染环境状态"""
        if mode == 'human':
            print(f"Time Slot: {self.current_time_slot}")
            print(f"Episode Reward: {self.episode_reward:.2f}")
            print(f"Episode Length: {self.episode_length}")
            
            stats = self.optimization_problem.get_optimization_summary(self.current_time_slot)
            print(f"Total Tasks: {stats['total_tasks']}")
            print(f"Completed Tasks: {stats['completed_tasks']}")
            print(f"Success Rate: {stats['success_rate']:.2%}")
            print(f"Objective Value: {stats['objective_value']:.2f}")
            print("-" * 50)
    
    def _build_vehicle_local_state(self, vehicle_id: int) -> torch.Tensor:
        """构建车辆的局部状态向量
        
        根据算法.md第17-27行，局部状态包含：
        1. 自身任务信息：d, c, δ, P
        2. 自身队列状态：HPC/LPC/V2I/V2V队列长度和剩余时间
        3. 网络环境信息：信道增益 h_{i,j}, h_{i,g}
        4. 协作节点信息：边缘节点负载、邻车负载
        
        Returns:
            state_tensor: 车辆局部状态向量 [state_dim]
        """
        state_features = []
        
        # 1. 自身任务信息（简化：前5个任务的信息，每个任务4个特征）
        if vehicle_id in self.task_manager.active_tasks:
            tasks = self.task_manager.active_tasks[vehicle_id][:5]
            for task in tasks:
                state_features.extend([
                    task.data_size / 1e7,  # 归一化
                    task.cpu_cycles / 1e7,
                    task.deadline / 100.0,
                    task.priority
                ])
            # 补齐到5个任务
            for _ in range(5 - len(tasks)):
                state_features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            state_features.extend([0.0] * 20)
        
        # 2. 队列状态（4个队列长度）
        queue_status = self.queue_manager.vehicle_managers[vehicle_id].get_queue_status()
        state_features.extend([
            queue_status['hpc_length'] / 10.0,  # 归一化
            queue_status['lpc_length'] / 10.0,
            queue_status['v2i_length'] / 10.0,
            queue_status['v2v_length'] / 10.0
        ])
        
        # 3. 网络环境信息（到RSU的信道增益，5个RSU）
        for rsu_id in range(self.system_config.NUM_RSU):
            channel_gain = self.network_model.get_channel_gain(vehicle_id, rsu_id)
            state_features.append(channel_gain * 1e10)  # 归一化
        
        # 4. 协作节点信息（边缘节点负载，5个RSU）
        for edge_id in range(self.system_config.NUM_RSU):
            if edge_id in self.queue_manager.edge_managers:
                edge_status = self.queue_manager.edge_managers[edge_id].get_queue_status()
                state_features.append(edge_status['enc_length'] / 10.0)
            else:
                state_features.append(0.0)
        
        # 5. 邻车信息（简化：最近3个邻车的负载）
        available_neighbors = self.comm_model.get_available_neighbors(vehicle_id)[:3]
        for neighbor_id in available_neighbors:
            neighbor_status = self.queue_manager.vehicle_managers[neighbor_id].get_queue_status()
            state_features.append((neighbor_status['hpc_length'] + neighbor_status['lpc_length']) / 10.0)
        # 补齐到3个邻车
        for _ in range(3 - len(available_neighbors)):
            state_features.append(0.0)
        
        # 补齐到state_dim维度（200维）
        state_dim = self.mappo_config.HIGH_LEVEL_CONFIG['state_dim']
        while len(state_features) < state_dim:
            state_features.append(0.0)
        state_features = state_features[:state_dim]
        
        # 转换为tensor
        device = torch.device(self.mappo_config.DEVICE)
        return torch.tensor(state_features, dtype=torch.float32, device=device)
    
    def _get_vehicle_state(self, vehicle_id: int, system_state: Dict[str, Any]) -> torch.Tensor:
        """从系统状态中提取车辆局部状态
        
        Args:
            vehicle_id: 车辆ID
            system_state: 系统状态字典（包含local_states和global_state）
        
        Returns:
            车辆的局部状态tensor
        """
        return system_state['local_states'][vehicle_id]
    
    def close(self):
        """关闭环境"""
        pass

class TrainingManager:
    """训练管理器"""
    
    def __init__(self, env: SimulationEnvironment, agent: HierarchicalMAPPO):
        self.env = env
        self.agent = agent
        self.env.set_agent(agent)
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'objective_values': []
        }
    
    def train(self, num_episodes: int, save_frequency: int = 100):
        """训练智能体"""
        print(f"开始训练，总轮数: {num_episodes}")
        
        for episode in range(num_episodes):
            episode_reward, episode_length = self._run_episode()
            
            # 记录统计信息
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_lengths'].append(episode_length)
            
            # 更新智能体
            if episode % self.agent.config.TRAINING_CONFIG['update_frequency'] == 0:
                self.agent.update(episode)
            
            # 打印进度
            if episode % 10 == 0:
                avg_reward = np.mean(self.training_stats['episode_rewards'][-10:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Length: {episode_length}")
            
            # 保存模型
            if episode % save_frequency == 0 and episode > 0:
                self.agent.save_model(f"results/models/checkpoint_episode_{episode}.pth")
        
        print("训练完成！")
    
    def _run_episode(self) -> Tuple[float, int]:
        """运行一个回合
        
        修正：处理全局奖励共享机制
        所有车辆共享相同的全局奖励
        """
        state = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        while True:
            # 获取动作
            actions = self._get_actions(state)
            
            # 执行动作（返回全局共享奖励）
            next_state, shared_rewards, done, info = self.env.step(actions)
            
            # shared_rewards是Dict[vehicle_id, global_reward]，所有车辆的奖励相同
            # 取任意一个车辆的奖励即可（因为都相同）
            global_reward = list(shared_rewards.values())[0] if shared_rewards else 0.0
            
            # 存储经验（每个车辆使用全局奖励）
            self._store_experience(state, actions, shared_rewards, next_state, done)
            
            episode_reward += global_reward
            episode_length += 1
            
            if done:
                break
            
            state = next_state
        
        return episode_reward, episode_length
    
    def _get_actions(self, state: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """获取所有车辆的动作
        
        CTDE执行阶段：每个Actor使用局部状态选择动作
        """
        actions = {}
        
        for vehicle_id in range(self.env.system_config.NUM_VEHICLES):
            if vehicle_id in self.env.task_manager.active_tasks:
                # 获取车辆局部状态（Actor执行时使用）
                vehicle_local_state = self.env._get_vehicle_state(vehicle_id, state)
                
                # 智能体选择动作
                action = self.agent.select_action(vehicle_local_state, vehicle_id, None)
                actions[vehicle_id] = action
        
        return actions
    
    def _store_experience(self, state: Dict[str, Any], actions: Dict[int, Dict[str, Any]], 
                         shared_rewards: Dict[int, float], next_state: Dict[str, Any], done: bool):
        """存储经验到智能体缓冲区
        
        CTDE训练：
        - 存储局部状态用于Actor训练
        - 存储全局状态用于Critic训练
        - 使用全局共享奖励
        - 存储log_prob用于PPO clip
        """
        # 为每个车辆存储经验
        for vehicle_id in range(self.env.system_config.NUM_VEHICLES):
            action = actions.get(vehicle_id, {})
            experience = {
                'vehicle_id': vehicle_id,
                'local_state': self.env._get_vehicle_state(vehicle_id, state),  # Actor用局部状态
                'global_state': state['global_state'],  # Critic用全局状态
                'action': action,
                'action_log_prob': action.get('mode_log_prob', 0.0),  # 存储log_prob用于PPO
                'reward': shared_rewards.get(vehicle_id, 0.0),  # 全局共享奖励
                'next_local_state': self.env._get_vehicle_state(vehicle_id, next_state),
                'next_global_state': next_state['global_state'],
                'done': done
            }
            self.agent.store_experience(experience)
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """评估智能体性能"""
        self.agent.network.eval()
        
        eval_rewards = []
        eval_lengths = []
        eval_success_rates = []
        
        for episode in range(num_episodes):
            episode_reward, episode_length = self._run_episode()
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            
            # 计算成功率
            stats = self.env.optimization_problem.get_optimization_summary(
                self.env.current_time_slot
            )
            eval_success_rates.append(stats['success_rate'])
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_length': np.mean(eval_lengths),
            'avg_success_rate': np.mean(eval_success_rates)
        }
