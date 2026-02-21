"""
MAPPO算法配置参数
基于分层MAPPO框架设计
"""

class MAPPOConfig:
    """MAPPO算法配置"""
    
    # 网络结构参数
    HIDDEN_SIZE = 256  # 隐藏层大小
    NUM_HIDDEN_LAYERS = 2  # 隐藏层数量
    ACTIVATION = 'tanh'  # 激活函数
    
    # 高层策略网络参数
    HIGH_LEVEL_CONFIG = {
        'state_dim': 200,  # 扩展状态维度：包含队列状态、网络环境、邻车信息等丰富状态
        'action_dim': 2,  # 动作维度：任务划分比（连续）+ 卸载方式（离散）
        'hidden_size': 256,
        'num_layers': 3,
        'learning_rate': 3e-4,
        'clip_ratio': 0.2,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'use_attention': True,  # 启用注意力机制
        'attention_config': {   # 注意力机制配置
            'hidden_dim': 128,
            'num_heads': 8,
            'output_dim': 100  # 注意力输出维度（减少冗余特征）
        }
    }
    
    # 低层策略网络参数
    LOW_LEVEL_CONFIG = {
        'v2i_scheduler': {
            'state_dim': 200,  # 扩展状态维度：与高层一致
            'action_dim': 6,  # 动作维度：目标RSU选择（离散）+ 发射功率（连续）
            'hidden_size': 128,
            'num_layers': 3,
            'learning_rate': 1e-4,  # 降低学习率，避免与高层冲突
            'entropy_coef': 0.01,  # 熵正则化系数
            'value_loss_coef': 0.5,  # 价值损失系数
        },
        'v2v_scheduler': {
            'state_dim': 200,  # 扩展状态维度：与高层一致
            'action_dim': 6,  # 动作维度：目标邻车选择（离散）+ 发射功率（连续）
            'hidden_size': 128,
            'num_layers': 3,
            'learning_rate': 1e-4,  # 降低学习率，避免与高层冲突
            'entropy_coef': 0.01,  # 熵正则化系数
            'value_loss_coef': 0.5,  # 价值损失系数
        },
        'local_computing': {
            'state_dim': 200,  # 扩展状态维度：与高层一致
            'action_dim': 1,  # 动作维度：本地计算频率（连续）
            'hidden_size': 64,
            'num_layers': 2,
            'learning_rate': 1e-4,  # 降低学习率，避免与高层冲突
            'entropy_coef': 0.01,  # 熵正则化系数
            'value_loss_coef': 0.5,  # 价值损失系数
        }
    }
    
    # 训练参数
    TRAINING_CONFIG = {
        'num_episodes': 10000,  # 训练轮数
        'max_steps_per_episode': 200,  # 每轮最大步数
        'min_steps_per_episode': 100,   # 每轮最小步数 - 增加以获得更多训练样本
        'update_frequency': 10,  # 更新频率
        'batch_size': 128,  # 增大批次大小，提升训练稳定性
        'gamma': 0.99,  # 折扣因子
        'gae_lambda': 0.95,  # GAE参数
        'ppo_clip_epsilon': 0.2,  # PPO clip参数
        'max_grad_norm': 0.5,  # 梯度裁剪
        'use_gae': True,  # 是否使用GAE
        'use_linear_lr_decay': True,  # 是否使用线性学习率衰减
        'use_orthogonal_init': True,  # 是否使用正交初始化
        'lr_decay_start': 1000,  # 学习率衰减开始轮数
        'lr_decay_episodes': 8000,  # 学习率衰减持续轮数
    }
    
    # 经验回放参数
    BUFFER_CONFIG = {
        'buffer_size': 50000,  # 增大经验缓冲区大小，提升样本多样性
        'min_buffer_size': 5000,  # 增大最小缓冲区大小，确保训练稳定
    }
    
    # 评估参数
    EVAL_CONFIG = {
        'eval_frequency': 100,  # 评估频率
        'num_eval_episodes': 10,  # 评估轮数
        'eval_max_steps': 500,  # 评估最大步数
    }
    
    # 设备配置
    DEVICE = 'cuda' if True else 'cpu'  # 是否使用GPU
    
    # 日志配置
    LOG_CONFIG = {
        'log_dir': './results/logs',
        'save_frequency': 500,  # 模型保存频率
        'log_frequency': 10,  # 日志记录频率
        'use_tensorboard': True,  # 是否使用TensorBoard
    }