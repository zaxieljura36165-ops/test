"""
基于队列和优先级的云边端卸载框架
基于系统建模.md中的优先级框架定义
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from configs.system_config import SystemConfig
from src.models.task_model import Task
from src.models.network_model import NetworkModel
from src.models.communication_model import CommunicationModel

class PriorityCalculator:
    """优先级计算器，实现任务优先级计算"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.eta_c = config.PRIORITY_WEIGHTS['cost']
        self.eta_u = config.PRIORITY_WEIGHTS['urgency']
        self.eta_k = config.PRIORITY_WEIGHTS['criticality']
        
        # 时间边界参数（用于归一化）
        self.t_min = 0.01  # 最小时延
        self.t_max = 2.0   # 最大时延
        
    def calculate_priority(self, task: Task, time_now: float, 
                          network_model: NetworkModel, 
                          comm_model: CommunicationModel) -> float:
        """计算任务优先级
        
        根据文档公式：
        P_{i,k} = η_C * C_{i,k} + η_U * U_{i,k} + η_K * K_{i,k}
        """
        # 1. 计算时延代价 C_{i,k}
        cost_t = self._calculate_delay_cost(task, network_model, comm_model)
        c_normalized = self._normalize_cost(cost_t)
        
        # 2. 计算紧急度 U_{i,k}
        urgency = self._calculate_urgency(task, time_now)
        
        # 3. 计算任务关键度 K_{i,k}
        criticality = self._calculate_criticality(task)
        
        # 计算最终优先级
        priority = (self.eta_c * c_normalized + 
                   self.eta_u * urgency + 
                   self.eta_k * criticality)
        
        return np.clip(priority, 0, 1)  # 确保在[0,1]范围内
    
    def _calculate_delay_cost(self, task: Task, network_model: NetworkModel, 
                            comm_model: CommunicationModel) -> float:
        """计算时延代价
        
        根据文档公式：
        costT_{i,k} = (T_{i,k}^{loc} + T_{i,k,j}^{V2I} + Σ T_{i,k,g}^{V2V}) / (G_i + 2)
        """
        # 本地处理时延（简化估计）
        t_loc = task.cpu_cycles / (3e9 * 0.5)  # 假设50%计算资源
        
        # V2I时延（选择最佳RSU）
        t_v2i = self._estimate_v2i_delay(task, network_model, comm_model)
        
        # V2V时延（选择最佳邻车）
        t_v2v = self._estimate_v2v_delay(task, network_model, comm_model)
        
        # 可用邻车数量
        available_neighbors = comm_model.get_available_neighbors(task.vehicle_id)
        g_i = len(available_neighbors)
        
        # 计算平均时延代价
        cost_t = (t_loc + t_v2i + t_v2v) / (g_i + 2)
        
        return cost_t
    
    def _estimate_v2i_delay(self, task: Task, network_model: NetworkModel, 
                           comm_model: CommunicationModel) -> float:
        """估计V2I时延"""
        min_delay = float('inf')
        
        for j in range(network_model.num_edge_nodes):
            # 检查是否在通信范围内
            if j in network_model.get_communication_range(task.vehicle_id):
                # 计算传输时延
                data_rate = comm_model.calculate_v2i_data_rate(
                    task.vehicle_id, j, self.config.MAX_TRANSMIT_POWER
                )
                if data_rate > 0:
                    transmission_delay = task.data_size / data_rate
                    # 计算处理时延（简化）
                    processing_delay = task.cpu_cycles / network_model.config.RSU_FREQ
                    total_delay = transmission_delay + processing_delay
                    min_delay = min(min_delay, total_delay)
        
        return min_delay if min_delay != float('inf') else 1.0
    
    def _estimate_v2v_delay(self, task: Task, network_model: NetworkModel, 
                           comm_model: CommunicationModel) -> float:
        """估计V2V时延"""
        min_delay = float('inf')
        available_neighbors = comm_model.get_available_neighbors(task.vehicle_id)
        
        for g in available_neighbors:
            # 计算传输时延
            data_rate = comm_model.calculate_v2v_data_rate(
                task.vehicle_id, g, self.config.MAX_TRANSMIT_POWER
            )
            if data_rate > 0:
                transmission_delay = task.data_size / data_rate
                # 计算处理时延（简化）
                processing_delay = task.cpu_cycles / (3e9 * 0.3)  # 假设30%计算资源
                total_delay = transmission_delay + processing_delay
                min_delay = min(min_delay, total_delay)
        
        return min_delay if min_delay != float('inf') else 1.0
    
    def _normalize_cost(self, cost_t: float) -> float:
        """归一化时延代价
        
        根据文档公式：
        C_{i,k} = clip((costT_{i,k} - t_min) / (t_max - t_min), 0, 1)
        """
        normalized = (cost_t - self.t_min) / (self.t_max - self.t_min)
        return np.clip(normalized, 0, 1)
    
    def _calculate_urgency(self, task: Task, time_now: float) -> float:
        """计算紧急度
        
        根据文档公式：
        U_{i,k} = clip(1 - t_slack / δ_i^max, 0, 1)
        t_slack = (arrival_time + deadline) - t_now
        """
        time_slack = (task.arrival_time + task.deadline) - time_now
        urgency = 1 - time_slack / self.config.TASK_DEADLINE_MAX
        return np.clip(urgency, 0, 1)
    
    def _calculate_criticality(self, task: Task) -> float:
        """计算任务关键度
        
        根据任务类型映射到[0,1]范围
        """
        criticality_map = {
            "safety": 0.9,      # 安全关键任务
            "emergency": 0.8,   # 紧急任务
            "normal": 0.3,      # 普通任务
            "entertainment": 0.1 # 娱乐任务
        }
        return criticality_map.get(task.task_type, 0.3)

class OffloadingDecisionMaker:
    """卸载决策制定器"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
    def make_offloading_decision(self, task: Task, alpha: float, 
                                available_nodes: Dict[str, List[int]]) -> Dict:
        """制定卸载决策
        
        根据任务划分比α和可用节点制定卸载决策
        """
        decision = {
            'alpha': alpha,
            'offload_mode': None,
            'target_node': None,
            'transmit_power': 0.0,
            'local_freq': 0.0
        }
        
        if alpha == 0:
            # 完全本地处理
            decision['local_freq'] = self.config.VEHICLE_MAX_FREQ * 0.8
        elif alpha == 1:
            # 完全卸载
            decision.update(self._choose_best_offload_target(task, available_nodes))
        else:
            # 部分卸载
            decision['local_freq'] = self.config.VEHICLE_MAX_FREQ * (1 - alpha)
            decision.update(self._choose_best_offload_target(task, available_nodes))
        
        return decision
    
    def _choose_best_offload_target(self, task: Task, 
                                   available_nodes: Dict[str, List[int]]) -> Dict:
        """选择最佳卸载目标"""
        best_decision = {
            'offload_mode': None,
            'target_node': None,
            'transmit_power': 0.0
        }
        
        best_score = -1
        
        # 评估V2I选项
        for rsu_id in available_nodes.get('V2I', []):
            score = self._evaluate_v2i_option(task, rsu_id)
            if score > best_score:
                best_score = score
                best_decision = {
                    'offload_mode': 'V2I',
                    'target_node': rsu_id,
                    'transmit_power': self.config.MAX_TRANSMIT_POWER * 0.8
                }
        
        # 评估V2V选项
        for neighbor_id in available_nodes.get('V2V', []):
            score = self._evaluate_v2v_option(task, neighbor_id)
            if score > best_score:
                best_score = score
                best_decision = {
                    'offload_mode': 'V2V',
                    'target_node': neighbor_id,
                    'transmit_power': self.config.MAX_TRANSMIT_POWER * 0.6
                }
        
        return best_decision
    
    def _evaluate_v2i_option(self, task: Task, rsu_id: int) -> float:
        """评估V2I选项的得分"""
        # 简化评估：基于任务优先级和RSU负载
        base_score = task.priority * 0.5
        # 这里可以加入RSU负载信息
        return base_score
    
    def _evaluate_v2v_option(self, task: Task, neighbor_id: int) -> float:
        """评估V2V选项的得分"""
        # 简化评估：基于任务优先级和邻车可用性
        base_score = task.priority * 0.3
        # 这里可以加入邻车负载信息
        return base_score

class PriorityFramework:
    """优先级框架主类"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.priority_calculator = PriorityCalculator(config)
        self.decision_maker = OffloadingDecisionMaker(config)
        
    def process_tasks(self, tasks: List[Task], time_now: float,
                     network_model: NetworkModel, 
                     comm_model: CommunicationModel) -> List[Dict]:
        """处理任务列表，计算优先级并制定决策"""
        decisions = []
        
        for task in tasks:
            # 计算优先级
            priority = self.priority_calculator.calculate_priority(
                task, time_now, network_model, comm_model
            )
            task.priority = priority
            
            # 获取可用节点
            available_nodes = self._get_available_nodes(task, network_model, comm_model)
            
            # 制定卸载决策（这里简化，实际应该由RL智能体决定）
            alpha = self._decide_task_partition(task)
            decision = self.decision_maker.make_offloading_decision(
                task, alpha, available_nodes
            )
            
            decisions.append({
                'task': task,
                'priority': priority,
                'decision': decision
            })
        
        return decisions
    
    def _get_available_nodes(self, task: Task, network_model: NetworkModel, 
                           comm_model: CommunicationModel) -> Dict[str, List[int]]:
        """获取可用节点列表"""
        available_nodes = {
            'V2I': [],
            'V2V': []
        }
        
        # V2I可用节点
        v2i_nodes = network_model.get_communication_range(task.vehicle_id)
        available_nodes['V2I'] = v2i_nodes
        
        # V2V可用节点
        v2v_nodes = comm_model.get_available_neighbors(task.vehicle_id)
        available_nodes['V2V'] = v2v_nodes
        
        return available_nodes
    
    def _decide_task_partition(self, task: Task) -> float:
        """决定任务划分比（简化版本，实际应该由RL智能体决定）"""
        # 基于任务优先级和类型决定划分比
        if task.priority > 0.7:  # 高优先级任务
            return 0.2  # 大部分本地处理
        elif task.priority > 0.4:  # 中等优先级任务
            return 0.5  # 平衡处理
        else:  # 低优先级任务
            return 0.8  # 大部分卸载
