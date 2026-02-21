"""
MADDPG配置文件
"""


class MADDPGConfig:
    """MADDPG算法配置"""
    
    # 设备配置
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    
    # 网络配置
    ACTOR_HIDDEN_SIZES = [256, 128]
    CRITIC_HIDDEN_SIZES = [512, 256, 128]
    
    # 学习率
    ACTOR_LR = 1e-4
    CRITIC_LR = 1e-3
    
    # 训练配置
    TRAINING_CONFIG = {
        'num_episodes': 1000,  # 训练轮数
        'max_steps_per_episode': 200,  # 每轮最大步数
        'min_steps_per_episode': 100,  # 每轮最小步数
        'batch_size': 128,  # 批大小
        'gamma': 0.99,  # 折扣因子
        'tau': 0.01,  # 软更新系数
        'update_frequency': 1,  # 保留字段（未使用）
        'updates_per_step': 1,  # 每步更新次数（未使用）
        'update_every_steps': 5,  # 每隔多少步更新一次（与分层TD3对齐）
    }
    
    # 经验回放
    REPLAY_CONFIG = {
        'buffer_size': 100000,  # 缓冲区大小
        'warmup_steps': 1000,  # 预热步数（在此之前不更新）
    }
    
    # 探索配置
    EXPLORATION_CONFIG = {
        'noise_scale': 0.1,  # 初始噪声强度
        'noise_decay': 0.9999,  # 噪声衰减率
        'min_noise_scale': 0.01,  # 最小噪声强度
        'gumbel_temperature_start': 1.0,  # Gumbel-Softmax初始温度
        'gumbel_temperature_end': 0.5,  # Gumbel-Softmax最终温度
        'gumbel_temperature_decay': 0.9995,  # 温度衰减率
    }
    
    # 评估配置
    EVAL_CONFIG = {
        'eval_frequency': 50,  # 每隔多少episode评估一次
        'num_eval_episodes': 5,  # 评估时运行多少个episode
    }
    
    # 日志配置
    LOG_CONFIG = {
        'log_dir': 'results/logs/maddpg',
        'model_dir': 'results/models/maddpg',
        'plot_dir': 'results/plots/maddpg',
    }
    
    @classmethod
    def get_config(cls) -> dict:
        """获取完整配置字典"""
        return {
            'device': cls.DEVICE,
            'actor_hidden_sizes': cls.ACTOR_HIDDEN_SIZES,
            'critic_hidden_sizes': cls.CRITIC_HIDDEN_SIZES,
            'actor_lr': cls.ACTOR_LR,
            'critic_lr': cls.CRITIC_LR,
            **cls.TRAINING_CONFIG,
            **cls.REPLAY_CONFIG,
            **cls.EXPLORATION_CONFIG,
        }

