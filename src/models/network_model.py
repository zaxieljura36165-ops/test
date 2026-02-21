"""
网络模型实现
基于系统建模.md中的网络模型定义
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
from configs.system_config import SystemConfig

class NetworkModel:
    """网络模型类，实现车辆、边缘节点和通信链路建模"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.num_vehicles = config.NUM_VEHICLES
        self.num_rsu = config.NUM_RSU
        self.num_mbs = config.NUM_MBS
        self.num_edge_nodes = self.num_rsu + self.num_mbs
        
        # 初始化车辆位置（随机分布）
        self.vehicle_positions = self._initialize_vehicle_positions()
        
        # 初始化边缘节点位置（固定分布）
        self.edge_positions = self._initialize_edge_positions()
        
        # 初始化信道增益矩阵
        self.channel_gains = self._initialize_channel_gains()
        
    def _initialize_vehicle_positions(self) -> np.ndarray:
        """初始化车辆位置，随机分布在1000m x 1000m区域内"""
        # 移除固定种子，允许每次episode有不同的初始位置
        positions = np.random.uniform(0, 1000, (self.num_vehicles, 2))
        return positions
    
    def _initialize_edge_positions(self) -> np.ndarray:
        """初始化边缘节点位置，RSU均匀分布，MBS在中心"""
        positions = np.zeros((self.num_edge_nodes, 2))
        
        # RSU位置：在区域内均匀分布
        for i in range(self.num_rsu):
            angle = 2 * np.pi * i / self.num_rsu
            radius = 400  # RSU分布半径
            positions[i] = [500 + radius * np.cos(angle), 
                          500 + radius * np.sin(angle)]
        
        # MBS位置：在中心
        positions[self.num_rsu] = [500, 500]
        
        return positions
    
    def _initialize_channel_gains(self) -> np.ndarray:
        """初始化信道增益矩阵 h_{i,j}"""
        # 形状: (num_vehicles, num_edge_nodes)
        channel_gains = np.zeros((self.num_vehicles, self.num_edge_nodes))
        
        for i in range(self.num_vehicles):
            for j in range(self.num_edge_nodes):
                # 计算距离
                distance = np.linalg.norm(self.vehicle_positions[i] - self.edge_positions[j])
                
                # 计算信道增益：h_{i,j} = G / d_{i,j}^β
                if distance > 0:
                    channel_gains[i, j] = (self.config.ANTENNA_GAIN / 
                                         (distance ** self.config.PATH_LOSS_EXPONENT))
                else:
                    channel_gains[i, j] = self.config.ANTENNA_GAIN  # 避免除零
                    
        return channel_gains
    
    def get_channel_gain(self, vehicle_id: int, edge_node_id: int) -> float:
        """获取车辆i到边缘节点j的信道增益"""
        return self.channel_gains[vehicle_id, edge_node_id]
    
    def get_distance(self, vehicle_id: int, edge_node_id: int) -> float:
        """获取车辆i到边缘节点j的距离"""
        return np.linalg.norm(self.vehicle_positions[vehicle_id] - 
                             self.edge_positions[edge_node_id])
    
    def update_vehicle_position(self, vehicle_id: int, new_position: np.ndarray):
        """更新车辆位置（用于动态场景）"""
        self.vehicle_positions[vehicle_id] = new_position
        # 重新计算该车辆到所有边缘节点的信道增益
        for j in range(self.num_edge_nodes):
            distance = np.linalg.norm(new_position - self.edge_positions[j])
            if distance > 0:
                self.channel_gains[vehicle_id, j] = (self.config.ANTENNA_GAIN / 
                                                   (distance ** self.config.PATH_LOSS_EXPONENT))
            else:
                self.channel_gains[vehicle_id, j] = self.config.ANTENNA_GAIN
    
    def get_communication_range(self, vehicle_id: int) -> List[int]:
        """获取车辆i的通信范围内的边缘节点列表"""
        available_nodes = []
        for j in range(self.num_edge_nodes):
            distance = self.get_distance(vehicle_id, j)
            # 假设通信范围为500m
            if distance <= 500:
                available_nodes.append(j)
        return available_nodes
    
    def calculate_data_rate(self, vehicle_id: int, edge_node_id: int, 
                          transmit_power: float) -> float:
        """计算车辆i到边缘节点j的数据传输速率
        
        根据Shannon公式：r_{i,j} = B * log2(1 + p_{i,k} * h_{i,j} / σ^2)
        """
        channel_gain = self.get_channel_gain(vehicle_id, edge_node_id)
        snr = (transmit_power * channel_gain) / self.config.NOISE_POWER
        
        # 避免log(0)的情况
        if snr <= 0:
            return 0.0
            
        data_rate = self.config.BANDWIDTH * np.log2(1 + snr)
        return data_rate
    
    def get_network_state(self) -> Dict:
        """获取当前网络状态"""
        return {
            'vehicle_positions': self.vehicle_positions.copy(),
            'edge_positions': self.edge_positions.copy(),
            'channel_gains': self.channel_gains.copy(),
            'num_vehicles': self.num_vehicles,
            'num_edge_nodes': self.num_edge_nodes
        }
    
    def reset(self):
        """重置网络模型"""
        self.vehicle_positions = self._initialize_vehicle_positions()
        self.channel_gains = self._initialize_channel_gains()
