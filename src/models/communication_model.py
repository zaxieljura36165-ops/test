"""
通信模型实现
基于系统建模.md中的通信模型定义
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from configs.system_config import SystemConfig
from src.models.network_model import NetworkModel

class CommunicationModel:
    """通信模型类，实现V2I和V2V通信建模"""
    
    def __init__(self, config: SystemConfig, network_model: NetworkModel):
        self.config = config
        self.network_model = network_model
        
        # V2V通信参数
        self.v2v_range = 200  # V2V通信范围 (m)
        self.v2v_bandwidth = config.BANDWIDTH * 0.5  # V2V带宽（假设为V2I的一半）
        
        # 初始化V2V信道增益矩阵
        self.v2v_channel_gains = self._initialize_v2v_channel_gains()
        
    def _initialize_v2v_channel_gains(self) -> np.ndarray:
        """初始化V2V信道增益矩阵 h_{i,g}"""
        # 形状: (num_vehicles, num_vehicles)
        v2v_gains = np.zeros((self.config.NUM_VEHICLES, self.config.NUM_VEHICLES))
        
        for i in range(self.config.NUM_VEHICLES):
            for g in range(self.config.NUM_VEHICLES):
                if i != g:  # 不计算自己到自己的信道增益
                    # 计算车辆间距离
                    distance = np.linalg.norm(
                        self.network_model.vehicle_positions[i] - 
                        self.network_model.vehicle_positions[g]
                    )
                    
                    # 计算V2V信道增益
                    if distance <= self.v2v_range and distance > 0:
                        v2v_gains[i, g] = (self.config.ANTENNA_GAIN / 
                                         (distance ** self.config.PATH_LOSS_EXPONENT))
                    else:
                        v2v_gains[i, g] = 0  # 超出通信范围
                        
        return v2v_gains
    
    def get_v2v_channel_gain(self, vehicle_i: int, vehicle_g: int) -> float:
        """获取车辆i到车辆g的V2V信道增益"""
        return self.v2v_channel_gains[vehicle_i, vehicle_g]
    
    def get_v2v_distance(self, vehicle_i: int, vehicle_g: int) -> float:
        """获取车辆i到车辆g的距离"""
        return np.linalg.norm(
            self.network_model.vehicle_positions[vehicle_i] - 
            self.network_model.vehicle_positions[vehicle_g]
        )
    
    def get_available_neighbors(self, vehicle_id: int) -> List[int]:
        """获取车辆i的可用邻车列表（在V2V通信范围内）"""
        neighbors = []
        for g in range(self.config.NUM_VEHICLES):
            if g != vehicle_id and self.v2v_channel_gains[vehicle_id, g] > 0:
                neighbors.append(g)
        return neighbors
    
    def calculate_v2i_data_rate(self, vehicle_id: int, edge_node_id: int, 
                               transmit_power: float) -> float:
        """计算V2I数据传输速率
        
        根据Shannon公式：r_{i,j}^{V2I} = B * log2(1 + p_{i,k} * h_{i,j} / σ^2)
        """
        channel_gain = self.network_model.get_channel_gain(vehicle_id, edge_node_id)
        snr = (transmit_power * channel_gain) / self.config.NOISE_POWER
        
        # 确保SNR为正值，避免数值问题
        snr = max(snr, 1e-10)  # 设置最小SNR值
        
        data_rate = self.config.BANDWIDTH * np.log2(1 + snr)
        return max(data_rate, 0.0)  # 确保非负
    
    def calculate_v2v_data_rate(self, vehicle_i: int, vehicle_g: int, 
                               transmit_power: float) -> float:
        """计算V2V数据传输速率
        
        根据Shannon公式：r_{i,g}^{V2V} = B_V2V * log2(1 + p_{i,k} * h_{i,g} / σ^2)
        """
        channel_gain = self.get_v2v_channel_gain(vehicle_i, vehicle_g)
        snr = (transmit_power * channel_gain) / self.config.NOISE_POWER
        
        # 确保SNR为正值，避免数值问题
        snr = max(snr, 1e-10)  # 设置最小SNR值
        
        data_rate = self.v2v_bandwidth * np.log2(1 + snr)
        return max(data_rate, 0.0)  # 确保非负
    
    def calculate_transmission_time(self, data_size: float, data_rate: float) -> float:
        """计算传输时间
        
        T = data_size / data_rate
        """
        if data_rate <= 0:
            return 1.0  # 返回一个合理的默认值而不是无穷大
        return data_size / data_rate
    
    def calculate_transmission_energy(self, transmit_power: float, 
                                    transmission_time: float) -> float:
        """计算传输能耗
        
        E = p * T
        """
        return transmit_power * transmission_time
    
    def update_v2v_channel_gains(self, vehicle_id: int, new_position: np.ndarray):
        """更新V2V信道增益（当车辆位置改变时）"""
        # 更新网络模型中的车辆位置
        self.network_model.update_vehicle_position(vehicle_id, new_position)
        
        # 重新计算该车辆到所有其他车辆的信道增益
        for g in range(self.config.NUM_VEHICLES):
            if g != vehicle_id:
                distance = self.get_v2v_distance(vehicle_id, g)
                if distance <= self.v2v_range and distance > 0:
                    self.v2v_channel_gains[vehicle_id, g] = (
                        self.config.ANTENNA_GAIN / 
                        (distance ** self.config.PATH_LOSS_EXPONENT)
                    )
                else:
                    self.v2v_channel_gains[vehicle_id, g] = 0
                    
                # 对称更新
                if distance <= self.v2v_range and distance > 0:
                    self.v2v_channel_gains[g, vehicle_id] = (
                        self.config.ANTENNA_GAIN / 
                        (distance ** self.config.PATH_LOSS_EXPONENT)
                    )
                else:
                    self.v2v_channel_gains[g, vehicle_id] = 0
    
    def get_communication_quality(self, vehicle_id: int, target_type: str, 
                                 target_id: int) -> Dict[str, float]:
        """获取通信质量指标"""
        if target_type == "V2I":
            channel_gain = self.network_model.get_channel_gain(vehicle_id, target_id)
            distance = self.network_model.get_distance(vehicle_id, target_id)
            max_data_rate = self.calculate_v2i_data_rate(
                vehicle_id, target_id, self.config.MAX_TRANSMIT_POWER
            )
        elif target_type == "V2V":
            channel_gain = self.get_v2v_channel_gain(vehicle_id, target_id)
            distance = self.get_v2v_distance(vehicle_id, target_id)
            max_data_rate = self.calculate_v2v_data_rate(
                vehicle_id, target_id, self.config.MAX_TRANSMIT_POWER
            )
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        return {
            'channel_gain': channel_gain,
            'distance': distance,
            'max_data_rate': max_data_rate,
            'snr_at_max_power': (self.config.MAX_TRANSMIT_POWER * channel_gain) / 
                               self.config.NOISE_POWER
        }
    
    def get_communication_state(self) -> Dict:
        """获取通信状态信息"""
        return {
            'v2v_channel_gains': self.v2v_channel_gains.copy(),
            'v2v_range': self.v2v_range,
            'v2v_bandwidth': self.v2v_bandwidth,
            'v2i_bandwidth': self.config.BANDWIDTH,
            'noise_power': self.config.NOISE_POWER
        }
    
    def reset(self):
        """重置通信模型"""
        self.v2v_channel_gains = self._initialize_v2v_channel_gains()
