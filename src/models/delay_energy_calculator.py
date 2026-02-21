"""
时延和能耗计算模块
基于系统建模.md中的时延和能耗计算公式
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from configs.system_config import SystemConfig
from src.models.task_model import Task
from src.models.network_model import NetworkModel
from src.models.communication_model import CommunicationModel
from src.models.queue_manager import GlobalQueueManager

class DelayEnergyCalculator:
    """时延和能耗计算器"""
    
    def __init__(self, config: SystemConfig, network_model: NetworkModel, 
                 comm_model: CommunicationModel, queue_manager: GlobalQueueManager):
        self.config = config
        self.network_model = network_model
        self.comm_model = comm_model
        self.queue_manager = queue_manager
    
    def _ensure_finite(self, value: float, default: float = 1.0) -> float:
        """确保数值是有限的"""
        if np.isnan(value) or np.isinf(value):
            return default
        # 限制数值范围，避免过大的值
        return np.clip(value, 0.0, 1000.0)  # 限制在0-1000范围内
        
    def calculate_local_delay(self, task: Task, time_now: float) -> Tuple[float, float]:
        """计算本地处理时延和能耗
        
        根据文档公式：
        T_{i,k}^{loc} = T_{i,k}^{locw} + T_{i,k}^{locc}
        E_{i,k}^{loc} = κ_i * (f_{i,k}^{loc})^2 * c_{i,k}^{loc}
        """
        # 1. 计算本地等待时间 T_{i,k}^{locw}
        waiting_time = self._calculate_local_waiting_time(task, time_now)
        
        # 2. 计算本地计算时间 T_{i,k}^{locc}
        computing_time = self._calculate_local_computing_time(task)
        
        # 3. 总时延
        total_delay = waiting_time + computing_time
        total_delay = self._ensure_finite(total_delay, 1.0)
        
        # 4. 计算能耗
        energy = self._calculate_local_energy(task)
        energy = self._ensure_finite(energy, 0.1)
        
        return total_delay, energy
    
    def _calculate_local_waiting_time(self, task: Task, time_now: float) -> float:
        """计算本地等待时间
        
        根据文档公式136：
        - 高优先级任务(P>=0.5)：只需等待HPC队列
        - 低优先级任务(P<0.5)：需等待HPC队列+LPC队列
        """
        vehicle_id = task.vehicle_id
        waiting_time = 0.0
        
        # HPC队列等待时间
        hpc_waiting = self.queue_manager.vehicle_managers[vehicle_id].hpc_queue.get_waiting_time(task)
        
        # 根据任务优先级决定等待时间
        if task.priority >= 0.5:
            # 高优先级任务：只等待HPC队列
            waiting_time = hpc_waiting
        else:
            # 低优先级任务：等待HPC队列 + LPC队列
            lpc_waiting = self.queue_manager.vehicle_managers[vehicle_id].lpc_queue.get_waiting_time(task)
            waiting_time = hpc_waiting + lpc_waiting
        
        return waiting_time
    
    def _calculate_local_computing_time(self, task: Task) -> float:
        """计算本地计算时间
        
        T_{i,k}^{locc} = c_{i,k}^{loc} / f_{i,k}^{loc}
        """
        if task.local_freq > 0:
            return task.local_cpu_cycles / task.local_freq
        else:
            # 使用默认计算频率
            default_freq = self.config.VEHICLE_MAX_FREQ * 0.5
            return task.local_cpu_cycles / max(default_freq, 1e6)  # 确保不为零
    
    def _calculate_local_energy(self, task: Task) -> float:
        """计算本地计算能耗
        
        E_{i,k}^{loc} = κ_i * (f_{i,k}^{loc})^2 * c_{i,k}^{loc}
        """
        if task.local_freq > 0:
            return self.config.VEHICLE_ENERGY_COEFF * (task.local_freq ** 2) * task.local_cpu_cycles
        else:
            default_freq = self.config.VEHICLE_MAX_FREQ * 0.5
            return self.config.VEHICLE_ENERGY_COEFF * (default_freq ** 2) * task.local_cpu_cycles
    
    def calculate_v2i_delay(self, task: Task, edge_node_id: int, time_now: float) -> Tuple[float, float]:
        """计算V2I卸载时延和能耗
        
        根据文档公式：
        T_{i,k,j}^{V2I} = T_{i,k}^{V2Iw} + T_{i,k,j}^{V2It} + T_{i,k,j}^{ENCw} + T_{i,k,j}^{ENCc}
        E_{i,k,j}^{V2I} = p_{i,k} * d_{i,k}^{offload} / r_{i,k,j}^{V2I}
        """
        # 1. V2I调度队列等待时间 T_{i,k}^{V2Iw}
        v2i_waiting = self._calculate_v2i_waiting_time(task, time_now)
        
        # 2. 传输时间 T_{i,k,j}^{V2It}
        transmission_time = self._calculate_v2i_transmission_time(task, edge_node_id)
        
        # 3. 边缘计算队列等待时间 T_{i,k,j}^{ENCw}
        enc_waiting = self._calculate_enc_waiting_time(task, edge_node_id)
        
        # 4. 边缘计算时间 T_{i,k,j}^{ENCc}
        enc_computing = self._calculate_enc_computing_time(task, edge_node_id)
        
        # 总时延
        total_delay = v2i_waiting + transmission_time + enc_waiting + enc_computing
        
        # 计算能耗
        energy = self._calculate_v2i_energy(task, edge_node_id, transmission_time)
        
        return total_delay, energy
    
    def _calculate_v2i_waiting_time(self, task: Task, time_now: float) -> float:
        """计算V2I调度队列等待时间"""
        vehicle_id = task.vehicle_id
        return self.queue_manager.vehicle_managers[vehicle_id].v2i_queue.get_waiting_time(task)
    
    def _calculate_v2i_transmission_time(self, task: Task, edge_node_id: int) -> float:
        """计算V2I传输时间
        
        T_{i,k,j}^{V2It} = d_{i,k}^{offload} / r_{i,k,j}^{V2I}
        """
        data_rate = self.comm_model.calculate_v2i_data_rate(
            task.vehicle_id, edge_node_id, task.transmit_power
        )
        if data_rate > 0:
            return task.offload_data_size / data_rate
        else:
            return 1.0  # 返回合理的默认值
    
    def _calculate_enc_waiting_time(self, task: Task, edge_node_id: int) -> float:
        """计算边缘计算队列等待时间"""
        return self.queue_manager.edge_managers[edge_node_id].get_waiting_time_estimate(task)
    
    def _calculate_enc_computing_time(self, task: Task, edge_node_id: int) -> float:
        """计算边缘计算时间
        
        T_{i,k,j}^{ENCc} = c_{i,k}^{offload} / f_j^{edge}
        """
        if edge_node_id < self.config.NUM_RSU:
            computing_freq = self.config.RSU_FREQ
        else:  # MBS
            computing_freq = self.config.CLOUD_FREQ
        
        return task.offload_cpu_cycles / max(computing_freq, 1e6)  # 确保不为零
    
    def _calculate_v2i_energy(self, task: Task, edge_node_id: int, transmission_time: float) -> float:
        """计算V2I能耗
        
        E_{i,k,j}^{V2I} = p_{i,k} * d_{i,k}^{offload} / r_{i,k,j}^{V2I}
        """
        return task.transmit_power * transmission_time
    
    def calculate_v2v_delay(self, task: Task, neighbor_id: int, time_now: float) -> Tuple[float, float]:
        """计算V2V卸载时延和能耗
        
        根据文档公式：
        T_{i,k,i'}^{V2V} = T_{i,k}^{V2Vw} + T_{i,k,i'}^{V2Vt} + T_{i,k,i'}^{LPCw} + T_{i,k,i'}^{LPCc}
        E_{i,k,i'}^{V2V} = p_{i,k} * d_{i,k}^{offload} / r_{i,k,i'}^{V2V} + 
                           κ_i * (f_{i,k}^{neighbor})^2 * c_{i,k}^{offload}
        """
        # 1. V2V调度队列等待时间 T_{i,k}^{V2Vw}
        v2v_waiting = self._calculate_v2v_waiting_time(task, time_now)
        
        # 2. 传输时间 T_{i,k,i'}^{V2Vt}
        transmission_time = self._calculate_v2v_transmission_time(task, neighbor_id)
        
        # 3. 邻车计算队列等待时间 T_{i,k,i'}^{LPCw}
        lpc_waiting = self._calculate_neighbor_lpc_waiting_time(task, neighbor_id)
        
        # 4. 邻车计算时间 T_{i,k,i'}^{LPCc}
        lpc_computing = self._calculate_neighbor_lpc_computing_time(task, neighbor_id)
        
        # 总时延
        total_delay = v2v_waiting + transmission_time + lpc_waiting + lpc_computing
        
        # 计算能耗
        energy = self._calculate_v2v_energy(task, neighbor_id, transmission_time)
        
        return total_delay, energy
    
    def _calculate_v2v_waiting_time(self, task: Task, time_now: float) -> float:
        """计算V2V调度队列等待时间"""
        vehicle_id = task.vehicle_id
        return self.queue_manager.vehicle_managers[vehicle_id].v2v_queue.get_waiting_time(task)
    
    def _calculate_v2v_transmission_time(self, task: Task, neighbor_id: int) -> float:
        """计算V2V传输时间
        
        T_{i,k,i'}^{V2Vt} = d_{i,k}^{offload} / r_{i,k,i'}^{V2V}
        """
        data_rate = self.comm_model.calculate_v2v_data_rate(
            task.vehicle_id, neighbor_id, task.transmit_power
        )
        if data_rate > 0:
            return task.offload_data_size / data_rate
        else:
            return 1.0  # 返回合理的默认值
    
    def _calculate_neighbor_lpc_waiting_time(self, task: Task, neighbor_id: int) -> float:
        """计算邻车LPC队列等待时间"""
        return self.queue_manager.vehicle_managers[neighbor_id].get_waiting_time_estimate(task, "V2V")
    
    def _calculate_neighbor_lpc_computing_time(self, task: Task, neighbor_id: int) -> float:
        """计算邻车LPC计算时间
        
        T_{i,k,i'}^{LPCc} = c_{i,k}^{offload} / f_{i,k}^{neighbor}
        """
        # 假设邻车分配30%计算资源给V2V任务
        neighbor_freq = self.config.VEHICLE_MAX_FREQ * 0.3
        return task.offload_cpu_cycles / max(neighbor_freq, 1e6)  # 确保不为零
    
    def _calculate_v2v_energy(self, task: Task, neighbor_id: int, transmission_time: float) -> float:
        """计算V2V能耗
        
        E_{i,k,i'}^{V2V} = p_{i,k} * d_{i,k}^{offload} / r_{i,k,i'}^{V2V} + 
                           κ_i * (f_{i,k}^{neighbor})^2 * c_{i,k}^{offload}
        """
        # 传输能耗
        transmission_energy = task.transmit_power * transmission_time
        
        # 邻车计算能耗
        neighbor_freq = self.config.VEHICLE_MAX_FREQ * 0.3
        computing_energy = (self.config.VEHICLE_ENERGY_COEFF * 
                           (neighbor_freq ** 2) * task.offload_cpu_cycles)
        
        return transmission_energy + computing_energy
    
    def calculate_total_delay(self, task: Task, time_now: float) -> float:
        """计算任务总时延
        
        根据文档公式：
        T_{i,k}^{total} = max(T_{i,k}^{loc}, T_{i,k}^{offload})
        """
        # 本地处理时延
        local_delay, _ = self.calculate_local_delay(task, time_now)
        
        # 卸载处理时延
        offload_delay = 0.0
        if task.offload_mode == "V2I" and task.target_node is not None:
            offload_delay, _ = self.calculate_v2i_delay(task, task.target_node, time_now)
        elif task.offload_mode == "V2V" and task.target_node is not None:
            offload_delay, _ = self.calculate_v2v_delay(task, task.target_node, time_now)
        
        # 总时延为本地和卸载的最大值
        return max(local_delay, offload_delay)
    
    def calculate_total_energy(self, task: Task, time_now: float) -> float:
        """计算任务总能耗"""
        # 本地计算能耗
        _, local_energy = self.calculate_local_delay(task, time_now)
        
        # 卸载能耗
        offload_energy = 0.0
        if task.offload_mode == "V2I" and task.target_node is not None:
            _, offload_energy = self.calculate_v2i_delay(task, task.target_node, time_now)
        elif task.offload_mode == "V2V" and task.target_node is not None:
            _, offload_energy = self.calculate_v2v_delay(task, task.target_node, time_now)
        
        return local_energy + offload_energy
    
    def calculate_penalty(self, task: Task, total_delay: float) -> float:
        """计算超时惩罚
        
        根据文档公式：
        Penalty_{i,k} = μ * (T_{i,k}^{total} - δ_{i,k}) / δ_{i,k} * (1 + λ_p*P_{i,k} + λ_u*U_{i,k})
        """
        if total_delay <= task.deadline:
            return 0.0
        
        # 计算紧急度
        # t_slack应该是剩余可用时间，但这里用于惩罚计算，使用deadline作为最大时间范围
        time_slack = task.deadline
        urgency = max(0, 1 - time_slack / self.config.TASK_DEADLINE_MAX)
        
        # 计算惩罚，限制delay_excess的范围
        delay_excess = min((total_delay - task.deadline) / task.deadline, 10.0)  # 限制最大10倍
        penalty_multiplier = (1 + self.config.PRIORITY_PENALTY_WEIGHT * task.priority + 
                            self.config.URGENCY_PENALTY_WEIGHT * urgency)
        
        penalty = self.config.PENALTY_MULTIPLIER * delay_excess * penalty_multiplier
        
        # 限制惩罚值范围
        penalty = self._ensure_finite(penalty, 0.0)
        return max(0, min(penalty, 100.0))  # 限制最大惩罚为100
    
    def calculate_task_cost(self, task: Task, time_now: float) -> float:
        """计算任务总成本
        
        根据文档公式：
        C(w_{i,k}) = φ_1 * T_{i,k}^{total} + φ_2 * E_{i,k}^{total} + φ_3 * Penalty_{i,k}
        """
        # 计算总时延和能耗
        total_delay = self.calculate_total_delay(task, time_now)
        total_energy = self.calculate_total_energy(task, time_now)
        
        # 计算超时惩罚
        penalty = self.calculate_penalty(task, total_delay)
        
        # 计算总成本
        cost = (self.config.COST_WEIGHTS['delay'] * total_delay + 
                self.config.COST_WEIGHTS['energy'] * total_energy + 
                self.config.COST_WEIGHTS['penalty'] * penalty)
        
        # 确保成本值在合理范围内
        cost = self._ensure_finite(cost, 1.0)
        return max(0, min(cost, 1000.0))  # 限制最大成本为1000
