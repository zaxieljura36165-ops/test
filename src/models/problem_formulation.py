"""
问题建模和优化目标函数
基于系统建模.md中的问题建模定义
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from configs.system_config import SystemConfig
from src.models.task_model import Task, TaskManager
from src.models.network_model import NetworkModel
from src.models.communication_model import CommunicationModel
from src.models.queue_manager import GlobalQueueManager
from src.models.delay_energy_calculator import DelayEnergyCalculator

class OptimizationProblem:
    """优化问题建模类"""
    
    def __init__(self, config: SystemConfig, network_model: NetworkModel, 
                 comm_model: CommunicationModel, queue_manager: GlobalQueueManager,
                 task_manager: TaskManager):
        self.config = config
        self.network_model = network_model
        self.comm_model = comm_model
        self.queue_manager = queue_manager
        self.task_manager = task_manager
        
        # 时延能耗计算器
        self.delay_energy_calc = DelayEnergyCalculator(
            config, network_model, comm_model, queue_manager
        )
        
        # 决策变量
        self.decision_variables = {
            'alpha': {},  # 任务划分比 α_{i,k}
            'offload_mode': {},  # 卸载方式 m_{i,k}
            'target_node': {},  # 目标节点 j_{i,k}
            'transmit_power': {},  # 发射功率 p_{i,k}
            'local_freq': {}  # 本地计算频率 f_{i,k}^{loc}
        }
    
    def set_decision_variables(self, vehicle_id: int, task_id: int, 
                              alpha: float, offload_mode: str, 
                              target_node: int, transmit_power: float, 
                              local_freq: float):
        """设置决策变量"""
        key = (vehicle_id, task_id)
        self.decision_variables['alpha'][key] = alpha
        self.decision_variables['offload_mode'][key] = offload_mode
        self.decision_variables['target_node'][key] = target_node
        self.decision_variables['transmit_power'][key] = transmit_power
        self.decision_variables['local_freq'][key] = local_freq
    
    def get_decision_variables(self, vehicle_id: int, task_id: int) -> Dict:
        """获取决策变量"""
        key = (vehicle_id, task_id)
        return {
            'alpha': self.decision_variables['alpha'].get(key, 0.0),
            'offload_mode': self.decision_variables['offload_mode'].get(key, None),
            'target_node': self.decision_variables['target_node'].get(key, None),
            'transmit_power': self.decision_variables['transmit_power'].get(key, 0.0),
            'local_freq': self.decision_variables['local_freq'].get(key, 0.0)
        }
    
    def apply_decision_to_task(self, task: Task):
        """将决策应用到任务"""
        key = (task.vehicle_id, task.task_id)
        
        # 获取决策变量
        alpha = self.decision_variables['alpha'].get(key, 0.0)
        offload_mode = self.decision_variables['offload_mode'].get(key, None)
        target_node = self.decision_variables['target_node'].get(key, None)
        transmit_power = self.decision_variables['transmit_power'].get(key, 0.0)
        local_freq = self.decision_variables['local_freq'].get(key, 0.0)
        
        # 应用任务划分
        task.partition_task(alpha)
        
        # 设置卸载决策
        if offload_mode and target_node is not None:
            task.set_offload_decision(offload_mode, target_node, transmit_power, local_freq)
    
    def calculate_objective_function(self, time_slot: int) -> float:
        """计算目标函数值
        
        根据文档公式：
        min E_π[Σ_{t=0}^T Σ_{i∈I} Σ_{k∈K_i(t)} C(w_{i,k})]
        """
        total_cost = 0.0
        
        # 遍历所有车辆的所有任务
        for vehicle_id in range(self.config.NUM_VEHICLES):
            if vehicle_id in self.task_manager.active_tasks:
                for task in self.task_manager.active_tasks[vehicle_id]:
                    if task.status == "pending":
                        # 应用决策到任务
                        self.apply_decision_to_task(task)
                        
                        # 计算任务成本
                        time_now = time_slot * self.config.TIME_SLOT_DURATION
                        task_cost = self.delay_energy_calc.calculate_task_cost(task, time_now)
                        total_cost += task_cost
        
        return total_cost
    
    def check_constraints(self) -> Dict[str, bool]:
        """检查约束条件"""
        constraints = {
            'alpha_valid': True,
            'target_node_valid': True,
            'power_valid': True,
            'freq_valid': True,
            'edge_capacity_valid': True
        }
        
        # 检查任务划分比约束：α_{i,k} ∈ [0,1]
        for key, alpha in self.decision_variables['alpha'].items():
            if not (0 <= alpha <= 1):
                constraints['alpha_valid'] = False
                break
        
        # 检查目标节点约束：j_{i,k} ∈ J ∪ {J+1}
        max_edge_node = self.config.NUM_RSU + self.config.NUM_MBS - 1
        for key, target_node in self.decision_variables['target_node'].items():
            if target_node is not None and not (0 <= target_node <= max_edge_node):
                constraints['target_node_valid'] = False
                break
        
        # 检查功率约束：p_{i,k} ∈ [0, p_{i,k}^{max}]
        for key, power in self.decision_variables['transmit_power'].items():
            if not (0 <= power <= self.config.MAX_TRANSMIT_POWER):
                constraints['power_valid'] = False
                break
        
        # 检查频率约束：f_{i,k}^{loc} ∈ [0, f_i^{max}]
        for key, freq in self.decision_variables['local_freq'].items():
            if not (0 <= freq <= self.config.VEHICLE_MAX_FREQ):
                constraints['freq_valid'] = False
                break
        
        # 检查边缘节点容量约束
        constraints['edge_capacity_valid'] = self._check_edge_capacity_constraints()
        
        return constraints
    
    def _check_edge_capacity_constraints(self) -> bool:
        """检查边缘节点容量约束
        
        根据文档约束：
        Σ_{i,k∈Q_j^{ENC}(t)} c_{i,k}^{edge} / f_j^{edge} ≤ C_j^{max}
        """
        for node_id in range(self.config.NUM_RSU + self.config.NUM_MBS):
            # 获取边缘节点队列状态
            if node_id in self.queue_manager.edge_managers:
                edge_status = self.queue_manager.edge_managers[node_id].get_queue_status()
                
                # 计算当前负载
                current_load = edge_status['enc_length'] * 1e6 / self.config.RSU_FREQ  # 简化计算
                
                # 检查是否超过容量限制
                max_capacity = 1.0  # 假设最大容量为1.0
                if current_load > max_capacity:
                    return False
        
        return True
    
    def get_feasible_actions(self, vehicle_id: int, task: Task) -> Dict:
        """获取可行动作空间"""
        feasible_actions = {
            'alpha_range': (0.0, 1.0),
            'offload_modes': [],
            'target_nodes': {
                'V2I': [],
                'V2V': []
            },
            'power_range': (0.0, self.config.MAX_TRANSMIT_POWER),
            'freq_range': (0.0, self.config.VEHICLE_MAX_FREQ)
        }
        
        # 获取可用的V2I节点
        v2i_nodes = self.network_model.get_communication_range(vehicle_id)
        feasible_actions['target_nodes']['V2I'] = v2i_nodes
        
        # 获取可用的V2V节点
        v2v_nodes = self.comm_model.get_available_neighbors(vehicle_id)
        feasible_actions['target_nodes']['V2V'] = v2v_nodes
        
        # 确定可用的卸载方式
        if v2i_nodes:
            feasible_actions['offload_modes'].append('V2I')
        if v2v_nodes:
            feasible_actions['offload_modes'].append('V2V')
        
        return feasible_actions
    
    def calculate_reward(self, vehicle_id: int, task: Task, time_now: float) -> float:
        """计算奖励函数
        
        奖励 = -任务成本
        """
        # 应用决策到任务
        self.apply_decision_to_task(task)
        
        # 计算任务成本
        task_cost = self.delay_energy_calc.calculate_task_cost(task, time_now)
        
        # 返回负成本作为奖励
        return -task_cost
    
    def calculate_global_reward(self, time_slot: int) -> float:
        """计算全局奖励"""
        total_reward = 0.0
        
        for vehicle_id in range(self.config.NUM_VEHICLES):
            if vehicle_id in self.task_manager.active_tasks:
                for task in self.task_manager.active_tasks[vehicle_id]:
                    if task.status == "pending":
                        time_now = time_slot * self.config.TIME_SLOT_DURATION
                        reward = self.calculate_reward(vehicle_id, task, time_now)
                        total_reward += reward
        
        return total_reward
    
    def get_system_state(self, time_slot: int) -> Dict:
        """获取系统状态"""
        time_now = time_slot * self.config.TIME_SLOT_DURATION
        
        state = {
            'time_slot': time_slot,
            'time_now': time_now,
            'network_state': self.network_model.get_network_state(),
            'communication_state': self.comm_model.get_communication_state(),
            'queue_state': self.queue_manager.get_global_queue_status(),
            'task_statistics': self.task_manager.get_task_statistics(),
            'objective_value': self.calculate_objective_function(time_slot),
            'constraints_satisfied': self.check_constraints()
        }
        
        return state
    
    def reset_decision_variables(self):
        """重置决策变量"""
        for var_type in self.decision_variables:
            self.decision_variables[var_type].clear()
    
    def get_optimization_summary(self, time_slot: int) -> Dict:
        """获取优化摘要"""
        state = self.get_system_state(time_slot)
        
        summary = {
            'time_slot': time_slot,
            'total_tasks': sum(len(tasks) for tasks in self.task_manager.active_tasks.values()),
            'completed_tasks': self.task_manager.get_task_statistics()['completed_tasks'],
            'failed_tasks': self.task_manager.get_task_statistics()['failed_tasks'],
            'success_rate': self.task_manager.get_task_statistics()['success_rate'],
            'objective_value': state['objective_value'],
            'constraints_violated': not all(state['constraints_satisfied'].values()),
            'average_queue_length': self._calculate_average_queue_length(),
            'system_utilization': self._calculate_system_utilization()
        }
        
        return summary
    
    def _calculate_average_queue_length(self) -> float:
        """计算平均队列长度"""
        total_length = 0
        total_queues = 0
        
        # 车辆队列
        for manager in self.queue_manager.vehicle_managers.values():
            status = manager.get_queue_status()
            total_length += (status['hpc_length'] + status['lpc_length'] + 
                           status['v2i_length'] + status['v2v_length'])
            total_queues += 4
        
        # 边缘节点队列
        for manager in self.queue_manager.edge_managers.values():
            status = manager.get_queue_status()
            total_length += status['enc_length']
            total_queues += 1
        
        return total_length / total_queues if total_queues > 0 else 0
    
    def _calculate_system_utilization(self) -> float:
        """计算系统利用率"""
        # 简化计算：基于队列长度和任务数量
        total_tasks = sum(len(tasks) for tasks in self.task_manager.active_tasks.values())
        max_capacity = self.config.NUM_VEHICLES * 10  # 假设每辆车最大10个任务
        
        return min(total_tasks / max_capacity, 1.0)
