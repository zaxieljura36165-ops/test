"""
任务模型实现
基于系统建模.md中的任务模型定义
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from configs.system_config import SystemConfig
from dataclasses import dataclass

@dataclass
class Task:
    """任务类，表示单个计算任务"""
    task_id: int
    vehicle_id: int
    data_size: float  # d_{i,k} 数据量 (bits)
    cpu_cycles: float  # c_{i,k} CPU周期数
    deadline: float  # δ_{i,k} 截止时间 (s)
    arrival_time: float  # 到达时间
    priority: float = 0.0  # P_{i,k} 优先级
    task_type: str = "normal"  # 任务类型
    
    # 任务划分相关
    alpha: float = 0.0  # 任务划分比
    local_data_size: float = 0.0  # d_{i,k}^{loc}
    local_cpu_cycles: float = 0.0  # c_{i,k}^{loc}
    offload_data_size: float = 0.0  # d_{i,k}^{offload}
    offload_cpu_cycles: float = 0.0  # c_{i,k}^{offload}
    
    # 卸载决策
    offload_mode: Optional[str] = None  # "V2I" or "V2V"
    target_node: Optional[int] = None  # 目标节点ID
    transmit_power: float = 0.0  # 发射功率
    local_freq: float = 0.0  # 本地计算频率
    
    # 执行状态
    status: str = "pending"  # pending, processing, completed, failed
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    
    def update_priority(self, cost_weight: float, urgency_weight: float,
                       criticality_weight: float, time_now: float):
        """更新任务优先级 - 严格按照文档公式实现

        根据文档公式：
        1. 时延代价 C_{i,k} = clip((costT_{i,k} - t_min)/(t_max - t_min), 0, 1)
        2. 紧急度 U_{i,k} = clip(1 - t_slack/δ_i^max, 0, 1)
        3. 优先级 P_{i,k} = η_C * C_{i,k} + η_U * U_{i,k} + η_K * K_{i,k}
        """
        # 1. 计算时延代价 C_{i,k} - 公式(87-91)
        cost_t = self._calculate_cost_t(time_now)
        t_min, t_max = self._get_delay_bounds()  # 系统经验得到的时间上下界
        cost_normalized = np.clip((cost_t - t_min) / (t_max - t_min), 0, 1)

        # 2. 计算紧急度 U_{i,k} - 公式(96-101)
        # 剩余可用时间 = (到达时间 + 截止时间) - 当前时间
        time_slack = (self.arrival_time + self.deadline) - time_now
        urgency = np.clip(1 - time_slack / self.deadline, 0, 1)  # δ_i^max = δ_i,k

        # 3. 计算任务关键度 K_{i,k}
        criticality = 0.9 if self.task_type == "safety" else 0.3

        # 4. 计算优先级 P_{i,k} - 公式(108-111)
        self.priority = (cost_weight * cost_normalized +
                        urgency_weight * urgency +
                        criticality_weight * criticality)

    def _calculate_cost_t(self, time_now: float) -> float:
        """计算时延代价 costT_{i,k} - 公式(87)

        costT_{i,k} = (T_{i,k}^{loc} + T_{i,k,j}^{V2I} + ∑_{g=1}^{G_i} T_{i,k,g}^{V2V}) / (G_i + 2)
        """
        # 简化实现：取本地、V2I、V2V三种方案的平均时延
        local_delay = self.cpu_cycles / (3e9 * 0.5)  # 假设本地频率为3GHz的50%

        # V2I时延（假设到最近的RSU）
        v2i_delay = (self.offload_data_size / 1e6) + (self.offload_cpu_cycles / 4e9)  # 传输+计算

        # V2V时延（假设到邻车）
        v2v_delay = (self.offload_data_size / 0.5e6) + (self.offload_cpu_cycles / 2e9)  # 传输+计算

        # 平均时延代价
        cost_t = (local_delay + v2i_delay + v2v_delay) / 3

        return cost_t

    def _get_delay_bounds(self) -> tuple:
        """获取系统时延上下界 - 系统经验值"""
        # 这些值应该根据实际系统运行统计得到
        t_min = 0.01  # 最小预期时延 10ms
        t_max = 5.0   # 最大预期时延 5s
        return t_min, t_max
    
    def partition_task(self, alpha: float):
        """任务划分
        
        根据划分比α将任务分为本地和卸载两部分：
        d_{i,k}^{loc} = (1-α) * d_{i,k}
        c_{i,k}^{loc} = (1-α) * c_{i,k}
        d_{i,k}^{offload} = α * d_{i,k}
        c_{i,k}^{offload} = α * c_{i,k}
        """
        self.alpha = alpha
        self.local_data_size = (1 - alpha) * self.data_size
        self.local_cpu_cycles = (1 - alpha) * self.cpu_cycles
        self.offload_data_size = alpha * self.data_size
        self.offload_cpu_cycles = alpha * self.cpu_cycles
    
    def set_offload_decision(self, mode: str, target_node: int, 
                           transmit_power: float, local_freq: float):
        """设置卸载决策"""
        self.offload_mode = mode
        self.target_node = target_node
        self.transmit_power = transmit_power
        self.local_freq = local_freq

class TaskGenerator:
    """任务生成器，根据泊松分布生成任务"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.task_counter = 0
        
    def generate_tasks(self, vehicle_id: int, time_slot: int) -> List[Task]:
        """为车辆i在时隙t生成任务"""
        # 泊松分布生成任务数量
        num_tasks = np.random.poisson(self.config.TASK_ARRIVAL_RATE)
        tasks = []
        
        for _ in range(num_tasks):
            task = self._create_single_task(vehicle_id, time_slot)
            tasks.append(task)
            
        return tasks
    
    def _create_single_task(self, vehicle_id: int, time_slot: int) -> Task:
        """创建单个任务"""
        # 生成任务属性
        data_size = np.random.uniform(self.config.TASK_DATA_MIN, 
                                    self.config.TASK_DATA_MAX)
        cpu_cycles = np.random.uniform(self.config.TASK_CPU_MIN, 
                                     self.config.TASK_CPU_MAX)
        deadline = np.random.uniform(self.config.TASK_DEADLINE_MIN, 
                                   self.config.TASK_DEADLINE_MAX)
        
        # 随机确定任务类型
        task_type = np.random.choice(["safety", "normal"], p=[0.2, 0.8])
        
        task = Task(
            task_id=self.task_counter,
            vehicle_id=vehicle_id,
            data_size=data_size,
            cpu_cycles=cpu_cycles,
            deadline=deadline,
            arrival_time=time_slot * self.config.TIME_SLOT_DURATION,
            task_type=task_type
        )
        
        self.task_counter += 1
        return task

class TaskManager:
    """任务管理器，负责任务的生命周期管理"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.task_generator = TaskGenerator(config)
        self.active_tasks: Dict[int, List[Task]] = {}  # vehicle_id -> tasks
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        
    def generate_tasks_for_all_vehicles(self, time_slot: int) -> Dict[int, List[Task]]:
        """为所有车辆生成任务"""
        new_tasks = {}
        for vehicle_id in range(self.config.NUM_VEHICLES):
            tasks = self.task_generator.generate_tasks(vehicle_id, time_slot)
            if tasks:
                new_tasks[vehicle_id] = tasks
                if vehicle_id not in self.active_tasks:
                    self.active_tasks[vehicle_id] = []
                self.active_tasks[vehicle_id].extend(tasks)
        return new_tasks
    
    def update_task_priorities(self, time_now: float):
        """更新所有活跃任务的优先级"""
        for vehicle_id, tasks in self.active_tasks.items():
            for task in tasks:
                if task.status == "pending":
                    task.update_priority(
                        self.config.PRIORITY_WEIGHTS['cost'],
                        self.config.PRIORITY_WEIGHTS['urgency'],
                        self.config.PRIORITY_WEIGHTS['criticality'],
                        time_now
                    )
    
    def get_high_priority_tasks(self, vehicle_id: int) -> List[Task]:
        """获取车辆的高优先级任务（优先级 > 0.5）"""
        if vehicle_id not in self.active_tasks:
            return []
        return [task for task in self.active_tasks[vehicle_id] 
                if task.priority > 0.5 and task.status == "pending"]
    
    def get_low_priority_tasks(self, vehicle_id: int) -> List[Task]:
        """获取车辆的低优先级任务（优先级 <= 0.5）"""
        if vehicle_id not in self.active_tasks:
            return []
        return [task for task in self.active_tasks[vehicle_id] 
                if task.priority <= 0.5 and task.status == "pending"]
    
    def complete_task(self, task: Task, completion_time: float):
        """标记任务为完成"""
        task.status = "completed"
        task.completion_time = completion_time
        self.completed_tasks.append(task)
        
        # 从活跃任务中移除
        if task.vehicle_id in self.active_tasks:
            if task in self.active_tasks[task.vehicle_id]:
                self.active_tasks[task.vehicle_id].remove(task)
    
    def fail_task(self, task: Task, fail_time: float):
        """标记任务为失败"""
        task.status = "failed"
        task.completion_time = fail_time
        self.failed_tasks.append(task)
        
        # 从活跃任务中移除
        if task.vehicle_id in self.active_tasks:
            if task in self.active_tasks[task.vehicle_id]:
                self.active_tasks[task.vehicle_id].remove(task)
    
    def cleanup_expired_tasks(self, current_time: float) -> int:
        """清理过期任务
        
        Returns:
            expired_count: 过期任务数量
        """
        expired_count = 0
        for vehicle_id, tasks in self.active_tasks.items():
            expired_tasks = [task for task in tasks 
                           if current_time > task.arrival_time + task.deadline]
            for task in expired_tasks:
                self.fail_task(task, current_time)
                expired_count += 1
        return expired_count
    
    def get_task_statistics(self) -> Dict:
        """获取任务统计信息"""
        total_completed = len(self.completed_tasks)
        total_failed = len(self.failed_tasks)
        total_active = sum(len(tasks) for tasks in self.active_tasks.values())
        
        return {
            'completed_tasks': total_completed,
            'failed_tasks': total_failed,
            'active_tasks': total_active,
            'success_rate': total_completed / (total_completed + total_failed) 
                          if (total_completed + total_failed) > 0 else 0
        }
