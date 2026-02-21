"""
系统配置参数
基于系统建模.md文档中的参数定义
"""

class SystemConfig:
    """系统基础配置"""
    
    # 车辆参数
    NUM_VEHICLES = 10  # 车辆数量 i
    VEHICLE_MAX_FREQ = 5.0e9  # 车辆最大计算频率 f_i^max (Hz) - 提高处理能力
    VEHICLE_ENERGY_COEFF = 1e-27  # 能耗系数 κ_i
    
    # 边缘节点参数
    NUM_RSU = 5  # RSU数量 n
    NUM_MBS = 1  # MBS数量 (连接到云端)
    RSU_FREQ = 10.0e9  # RSU计算频率 f_j^edge (Hz) - 大幅提高边缘处理能力
    CLOUD_FREQ = 20.0e9  # 云服务器频率 f_cloud (Hz) - 大幅提高云端处理能力
    
    # 时隙参数
    TIME_SLOT_DURATION = 1.0  # 时隙长度 (s)
    NUM_TIME_SLOTS = 1000  # 总时隙数
    
    # 任务生成参数
    TASK_ARRIVAL_RATE = 0.2  # 泊松分布参数 λ 
    TASK_DATA_MIN = 0.5e6  # 最小数据量 d_i^min (bits) 
    TASK_DATA_MAX = 3e6  # 最大数据量 d_i^max (bits) 
    TASK_CPU_MIN = 0.5e6  # 最小CPU周期 c_i^min (cycles) 
    TASK_CPU_MAX = 3e6  # 最大CPU周期 c_i^max (cycles) 
    TASK_DEADLINE_MIN = 100.0  # 最小截止时间 δ_i^min (s) 
    TASK_DEADLINE_MAX = 200.0  # 最大截止时间 δ_i^max (s) 
    
    # 通信参数
    BANDWIDTH = 1e6  # 带宽 B (Hz)
    NOISE_POWER = 1e-13  # 噪声功率 σ^2 (W)
    PATH_LOSS_EXPONENT = 2.0  # 路径损耗指数 β
    ANTENNA_GAIN = 1.0  # 天线增益 G
    
    # 功率参数
    MAX_TRANSMIT_POWER = 1.0  # 最大发射功率 p_i,k^max (W)
    
    # 优先级权重
    PRIORITY_WEIGHTS = {
        'cost': 0.4,  # η_C
        'urgency': 0.4,  # η_U  
        'criticality': 0.2  # η_K
    }
    
    # 成本函数权重
    COST_WEIGHTS = {
        'delay': 0.1,  # φ_1 - 降低时延权重
        'energy': 0.01,  # φ_2 - 降低能耗权重
        'penalty': 1.0  # φ_3 - 大幅降低惩罚权重
    }
    
    # 超时惩罚参数
    PENALTY_MULTIPLIER = 0.01  # μ - 大幅降低惩罚倍数
    PRIORITY_PENALTY_WEIGHT = 0.1  # λ_p - 降低优先级惩罚权重
    URGENCY_PENALTY_WEIGHT = 0.1  # λ_u - 降低紧急度惩罚权重
