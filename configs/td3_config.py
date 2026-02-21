"""
TD3 (Twin Delayed DDPG) 算法配置
适配多车辆任务卸载场景

核心特性：
1. 双Q网络减少过估计
2. 延迟策略更新
3. 目标策略平滑正则化
4. 情境感知自适应噪声控制
"""

class TD3Config:
    """TD3算法配置"""
    
    # 设备配置
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    
    # ========== 网络架构配置 ==========
    HIGH_LEVEL_CONFIG = {
        'state_dim': 200,           # 局部状态维度
        'action_dim': 3,            # [alpha, mode_logit, power]
        'hidden_sizes': [256, 256, 128],
        'learning_rate': 3e-4,
        'use_layer_norm': True,     # 层归一化提高稳定性
    }
    
    LOW_LEVEL_CONFIG = {
        'v2i_scheduler': {
            'state_dim': 200,
            'action_dim': 6,        # [rsu_0_logit, ..., rsu_4_logit, power]
            'hidden_sizes': [128, 128],
            'learning_rate': 3e-4,
        },
        'v2v_scheduler': {
            'state_dim': 200,
            'action_dim': 6,        # [neighbor_0_logit, ..., neighbor_4_logit, power]
            'hidden_sizes': [128, 128],
            'learning_rate': 3e-4,
        },
        'local_computing': {
            'state_dim': 200,
            'action_dim': 1,        # [freq]
            'hidden_sizes': [64, 64],
            'learning_rate': 3e-4,
        }
    }
    
    # ========== TD3 核心参数 ==========
    TD3_CONFIG = {
        'gamma': 0.99,              # 折扣因子
        'tau': 0.005,               # 软更新系数
        'policy_delay': 2,          # 策略延迟更新频率
        'policy_noise': 0.2,        # 目标策略噪声
        'noise_clip': 0.5,          # 噪声裁剪范围
        'exploration_noise': 0.1,   # 探索噪声（初始值，会被噪声控制器覆盖）
    }
    
    # ========== 经验回放配置 ==========
    BUFFER_CONFIG = {
        'buffer_size': 50000,       # 总缓冲区大小（降低内存占用）
        'batch_size': 256,
        'min_buffer_size': 2000,    # 开始训练的最小样本数
        'use_dual_buffer': True,    # 使用双缓冲区
    }
    
    # ========== 噪声控制器配置 ==========
    NOISE_CONFIG = {
        # 阶段转换阈值
        'phase1_episodes': 100,         # Phase1随机探索轮数（增加到100）
        'phase2_min_episodes': 50,      # Phase2最小持续轮数，防止直接跳到Phase3
        'phase2_threshold': 0.5,        # 进入Phase2的成功率阈值
        'phase3_threshold': 0.9,        # 进入Phase3的成功率阈值（提高到90%）
        
        # 全局噪声强度 - 更缓和的噪声衰减
        'phase1_noise': 0.5,            # 阶段1噪声（降低初始噪声，从0.9降到0.5）
        'phase2_base_noise': 0.35,      # 阶段2基础噪声
        'phase3_start_noise': 0.25,     # 阶段3起始噪声（提高到0.25）
        'min_noise': 0.1,               # 最小噪声（提高到0.1，核心保底）
        'phase3_decay': 0.99995,        # 阶段3衰减率（更慢衰减）
        
        # 情境噪声因子（任务卸载场景）
        'context_noise_factors': {
            'high_load': 0.8,           # 高负载：较高噪声，探索更优卸载策略
            'low_load': 0.3,            # 低负载：低噪声，稳定执行
            'urgent_task': 0.4,         # 紧急任务：中低噪声，确保时效
            'normal_task': 0.6,         # 普通任务：中等噪声
            'congested_network': 0.7,   # 网络拥塞：较高噪声，探索替代方案
            'good_channel': 0.3,        # 良好信道：低噪声，充分利用
            'near_deadline': 0.3,       # 接近截止时间：低噪声，稳定决策
            'high_priority': 0.4,       # 高优先级任务：中低噪声
        },
        
        # 情境判断阈值
        'high_load_threshold': 0.7,     # 队列负载 > 70% 视为高负载
        'low_load_threshold': 0.3,      # 队列负载 < 30% 视为低负载
        'urgent_priority_threshold': 0.7,   # 优先级 > 0.7 视为紧急
        'near_deadline_ratio': 0.3,     # 剩余时间/截止时间 < 30% 视为接近截止
        'good_channel_threshold': 0.6,  # 信道质量 > 0.6 视为良好
    }
    
    # ========== 训练配置 ==========
    TRAINING_CONFIG = {
        'num_episodes': 2000,
        'max_steps_per_episode': 200,
        'min_steps_per_episode': 50,
        'update_frequency': 1,          # 每步更新
        'save_frequency': 100,
        'eval_frequency': 50,
        'num_eval_episodes': 10,
        
        # 学习率衰减
        'use_lr_decay': True,
        'lr_decay_start': 500,
        'lr_decay_episodes': 1000,
        'lr_min_ratio': 0.1,
        
        # 梯度裁剪
        'max_grad_norm': 0.5,
    }
    
    # ========== 双缓冲区采样比例 ==========
    DUAL_BUFFER_CONFIG = {
        'det_ratio_min': 0.1,           # 确定性经验最小比例
        'det_ratio_max': 0.4,           # 确定性经验最大比例
    }

