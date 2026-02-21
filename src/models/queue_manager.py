"""
HPC/LPC优先级队列管理系统
基于系统建模.md中的队列模型定义
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Deque
from collections import deque
from dataclasses import dataclass
from configs.system_config import SystemConfig
from src.models.task_model import Task

@dataclass
class QueueItem:
    """队列项，包含任务和相关信息"""
    task: Task
    arrival_time: float
    estimated_processing_time: float
    remaining_work: float  # 剩余工作量百分比 [0,1]
    
class PriorityQueue:
    """优先级队列基类"""
    
    def __init__(self, queue_name: str):
        self.queue_name = queue_name
        self.items: Deque[QueueItem] = deque()
        self.current_processing: Optional[QueueItem] = None
        self.total_processing_time = 0.0
        
    def add_task(self, task: Task, arrival_time: float, 
                estimated_processing_time: float):
        """添加任务到队列"""
        queue_item = QueueItem(
            task=task,
            arrival_time=arrival_time,
            estimated_processing_time=estimated_processing_time,
            remaining_work=1.0
        )
        self.items.append(queue_item)
        
    def get_queue_length(self) -> int:
        """获取队列长度"""
        return len(self.items)
    
    def is_empty(self) -> bool:
        """检查队列是否为空"""
        return len(self.items) == 0 and self.current_processing is None
    
    def get_next_task(self) -> Optional[QueueItem]:
        """获取下一个要处理的任务"""
        if self.current_processing is None and self.items:
            self.current_processing = self.items.popleft()
        return self.current_processing
    
    def complete_current_task(self):
        """完成当前正在处理的任务"""
        if self.current_processing:
            self.current_processing = None
    
    def update_processing_progress(self, delta_time: float, processing_rate: float):
        """更新处理进度
        
        Args:
            delta_time: 时间增量（秒）
            processing_rate: 处理速率（Hz/cycles per second），对于传输队列为0
        """
        if self.current_processing:
            # 计算完成的工作量百分比
            if processing_rate > 0:
                # 计算队列：基于estimated_processing_time（已经考虑了正确的cpu_cycles）
                # 注意：estimated_time在add_task时已经根据正确的cpu_cycles计算
                estimated_time = self.current_processing.estimated_processing_time
                if estimated_time > 0:
                    work_done_percentage = delta_time / estimated_time
                    self.current_processing.remaining_work = max(0, 
                        self.current_processing.remaining_work - work_done_percentage)
            else:
                # 传输队列：基于估计的传输时间
                estimated_time = self.current_processing.estimated_processing_time
                if estimated_time > 0:
                    work_done_percentage = delta_time / estimated_time
                    self.current_processing.remaining_work = max(0, 
                        self.current_processing.remaining_work - work_done_percentage)
            
            if self.current_processing.remaining_work <= 0:
                self.complete_current_task()
    
    def get_waiting_time(self, new_task: Task) -> float:
        """计算新任务需要等待的时间"""
        waiting_time = 0.0
        
        # 当前正在处理的任务剩余时间
        if self.current_processing:
            remaining_time = (self.current_processing.remaining_work * 
                            self.current_processing.estimated_processing_time)
            waiting_time += remaining_time
        
        # 队列中所有任务的处理时间
        for item in self.items:
            waiting_time += item.estimated_processing_time
            
        return waiting_time

class HPCQueue(PriorityQueue):
    """高优先级队列（HPC）"""
    
    def __init__(self):
        super().__init__("HPC")
        self.priority_threshold = 0.5  # 优先级阈值
        
    def add_task(self, task: Task, arrival_time: float, 
                estimated_processing_time: float):
        """添加高优先级任务"""
        if task.priority >= self.priority_threshold:
            super().add_task(task, arrival_time, estimated_processing_time)
        else:
            raise ValueError("Task priority too low for HPC queue")

class LPCQueue(PriorityQueue):
    """低优先级队列（LPC）"""
    
    def __init__(self):
        super().__init__("LPC")
        self.priority_threshold = 0.5  # 优先级阈值
        
    def add_task(self, task: Task, arrival_time: float, 
                estimated_processing_time: float):
        """添加低优先级任务"""
        if task.priority < self.priority_threshold:
            super().add_task(task, arrival_time, estimated_processing_time)
        else:
            raise ValueError("Task priority too high for LPC queue")

class VehicleQueueManager:
    """车辆队列管理器"""
    
    def __init__(self, vehicle_id: int, config: SystemConfig):
        self.vehicle_id = vehicle_id
        self.config = config
        
        # 本地计算队列
        self.hpc_queue = HPCQueue()  # 高优先级队列
        self.lpc_queue = LPCQueue()  # 低优先级队列
        
        # 卸载队列
        self.v2i_queue = PriorityQueue("V2I")  # V2I卸载队列
        self.v2v_queue = PriorityQueue("V2V")  # V2V卸载队列
        
        # 计算资源分配
        self.available_freq = config.VEHICLE_MAX_FREQ
        self.hpc_freq_ratio = 0.7  # HPC队列分配70%计算资源
        self.lpc_freq_ratio = 0.3  # LPC队列分配30%计算资源
        
    def add_local_task(self, task: Task, arrival_time: float):
        """添加本地任务到相应队列"""
        # 计算估计处理时间
        estimated_time = self._calculate_processing_time(task)
        
        # 根据优先级分配到相应队列
        if task.priority >= 0.5:
            self.hpc_queue.add_task(task, arrival_time, estimated_time)
        else:
            self.lpc_queue.add_task(task, arrival_time, estimated_time)
    
    def add_offload_task(self, task: Task, arrival_time: float, 
                        offload_mode: str):
        """添加卸载任务到相应队列"""
        estimated_time = self._calculate_transmission_time(task, offload_mode)
        
        if offload_mode == "V2I":
            self.v2i_queue.add_task(task, arrival_time, estimated_time)
        elif offload_mode == "V2V":
            self.v2v_queue.add_task(task, arrival_time, estimated_time)
    
    def _calculate_processing_time(self, task: Task) -> float:
        """计算本地处理时间"""
        if task.local_freq > 0:
            return task.local_cpu_cycles / task.local_freq
        else:
            # 使用默认频率计算本地部分
            return task.local_cpu_cycles / (self.available_freq * 0.5)
    
    def _calculate_transmission_time(self, task: Task, offload_mode: str) -> float:
        """计算传输时间（简化）"""
        if offload_mode == "V2I":
            # 简化：假设固定传输速率
            return task.offload_data_size / (1e6 * 0.5)  # 0.5 Mbps
        else:  # V2V
            return task.offload_data_size / (1e6 * 0.3)  # 0.3 Mbps
    
    def process_queues(self, delta_time: float):
        """处理所有队列"""
        # 首先处理HPC队列
        if not self.hpc_queue.is_empty():
            self._process_queue(self.hpc_queue, delta_time, 
                              self.available_freq * self.hpc_freq_ratio)
        
        # 只有当HPC队列为空时才处理LPC队列
        if self.hpc_queue.is_empty() and not self.lpc_queue.is_empty():
            self._process_queue(self.lpc_queue, delta_time, 
                              self.available_freq * self.lpc_freq_ratio)
        
        # 处理卸载队列
        self._process_queue(self.v2i_queue, delta_time, 0)  # 传输不需要本地计算
        self._process_queue(self.v2v_queue, delta_time, 0)
    
    def _process_queue(self, queue: PriorityQueue, delta_time: float, 
                      processing_rate: float):
        """处理单个队列"""
        if not queue.is_empty():
            queue.update_processing_progress(delta_time, processing_rate)
    
    def get_queue_status(self) -> Dict:
        """获取队列状态"""
        return {
            'hpc_length': self.hpc_queue.get_queue_length(),
            'lpc_length': self.lpc_queue.get_queue_length(),
            'v2i_length': self.v2i_queue.get_queue_length(),
            'v2v_length': self.v2v_queue.get_queue_length(),
            'hpc_processing': self.hpc_queue.current_processing is not None,
            'lpc_processing': self.lpc_queue.current_processing is not None
        }
    
    def get_waiting_time_estimate(self, task: Task, offload_mode: str = None) -> float:
        """获取任务等待时间估计"""
        if offload_mode is None:  # 本地处理
            if task.priority >= 0.5:
                return self.hpc_queue.get_waiting_time(task)
            else:
                # LPC队列需要等待HPC队列完成
                hpc_waiting = self.hpc_queue.get_waiting_time(task)
                lpc_waiting = self.lpc_queue.get_waiting_time(task)
                return hpc_waiting + lpc_waiting
        else:  # 卸载处理
            if offload_mode == "V2I":
                return self.v2i_queue.get_waiting_time(task)
            else:  # V2V
                # V2V接收方计算队列等待：HPC + LPC（文档公式181）
                # 注意：V2V任务总是进入邻车的LPC队列
                hpc_waiting = self.hpc_queue.get_waiting_time(task)
                lpc_waiting = self.lpc_queue.get_waiting_time(task)
                return hpc_waiting + lpc_waiting

class EdgeNodeQueueManager:
    """边缘节点队列管理器"""
    
    def __init__(self, node_id: int, node_type: str, config: SystemConfig):
        self.node_id = node_id
        self.node_type = node_type  # "RSU" or "MBS"
        self.config = config
        
        # 边缘计算队列
        self.enc_queue = PriorityQueue("ENC")
        
        # 计算频率
        if node_type == "RSU":
            self.computing_freq = config.RSU_FREQ
        else:  # MBS
            self.computing_freq = config.CLOUD_FREQ
    
    def add_task(self, task: Task, arrival_time: float):
        """添加任务到边缘节点队列"""
        estimated_time = task.offload_cpu_cycles / self.computing_freq
        self.enc_queue.add_task(task, arrival_time, estimated_time)
    
    def process_queue(self, delta_time: float):
        """处理队列"""
        if not self.enc_queue.is_empty():
            self.enc_queue.update_processing_progress(delta_time, self.computing_freq)
    
    def get_queue_status(self) -> Dict:
        """获取队列状态"""
        return {
            'enc_length': self.enc_queue.get_queue_length(),
            'enc_processing': self.enc_queue.current_processing is not None,
            'computing_freq': self.computing_freq
        }
    
    def get_waiting_time_estimate(self, task: Task) -> float:
        """获取任务等待时间估计"""
        return self.enc_queue.get_waiting_time(task)

class GlobalQueueManager:
    """全局队列管理器"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # 车辆队列管理器
        self.vehicle_managers = {
            i: VehicleQueueManager(i, config) 
            for i in range(config.NUM_VEHICLES)
        }
        
        # 边缘节点队列管理器
        self.edge_managers = {}
        
        # RSU队列管理器
        for i in range(config.NUM_RSU):
            self.edge_managers[i] = EdgeNodeQueueManager(i, "RSU", config)
        
        # MBS队列管理器
        mbs_id = config.NUM_RSU
        self.edge_managers[mbs_id] = EdgeNodeQueueManager(mbs_id, "MBS", config)
    
    def add_local_task(self, vehicle_id: int, task: Task, arrival_time: float):
        """添加本地任务"""
        self.vehicle_managers[vehicle_id].add_local_task(task, arrival_time)
    
    def add_offload_task(self, vehicle_id: int, task: Task, arrival_time: float, 
                        offload_mode: str):
        """添加卸载任务"""
        self.vehicle_managers[vehicle_id].add_offload_task(
            task, arrival_time, offload_mode
        )
    
    def add_edge_task(self, node_id: int, task: Task, arrival_time: float):
        """添加边缘节点任务"""
        self.edge_managers[node_id].add_task(task, arrival_time)
    
    def process_all_queues(self, delta_time: float):
        """处理所有队列"""
        # 处理车辆队列
        for manager in self.vehicle_managers.values():
            manager.process_queues(delta_time)
        
        # 处理边缘节点队列
        for manager in self.edge_managers.values():
            manager.process_queue(delta_time)
    
    def get_global_queue_status(self) -> Dict:
        """获取全局队列状态"""
        status = {
            'vehicles': {},
            'edge_nodes': {}
        }
        
        for vehicle_id, manager in self.vehicle_managers.items():
            status['vehicles'][vehicle_id] = manager.get_queue_status()
        
        for node_id, manager in self.edge_managers.items():
            status['edge_nodes'][node_id] = manager.get_queue_status()
        
        return status
    
    def get_waiting_time_estimate(self, task: Task, processing_type: str, 
                                 target_id: int) -> float:
        """获取等待时间估计"""
        if processing_type == "local":
            return self.vehicle_managers[task.vehicle_id].get_waiting_time_estimate(task)
        elif processing_type == "V2I":
            return self.edge_managers[target_id].get_waiting_time_estimate(task)
        elif processing_type == "V2V":
            return self.vehicle_managers[target_id].get_waiting_time_estimate(task, "V2V")
        else:
            return 0.0
