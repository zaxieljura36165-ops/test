"""
SAC (Soft Actor-Critic) configuration for standard (non-hierarchical) training.
"""


class SACConfig:
    """SAC algorithm configuration"""

    # Device
    DEVICE = "cuda"  # "cuda" or "cpu"

    # Network config (shared by actor/critic)
    HIGH_LEVEL_CONFIG = {
        "state_dim": 200,
        "hidden_sizes": [256, 256, 128],
        "learning_rate": 3e-4,
        "use_layer_norm": True,
    }

    # SAC core parameters
    SAC_CONFIG = {
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.2,           # entropy coefficient
        "auto_alpha": False,    # set True to auto-tune alpha
        "alpha_lr": 3e-4,
        "target_entropy": None,  # None -> -action_dim
        "log_std_min": -20,
        "log_std_max": 2,
    }

    # Replay buffer
    BUFFER_CONFIG = {
        "buffer_size": 50000,    # reduce memory usage
        "batch_size": 256,
        "min_buffer_size": 2000,
        "use_dual_buffer": True,
    }

    # Training config
    TRAINING_CONFIG = {
        "num_episodes": 2000,
        "max_steps_per_episode": 200,
        "min_steps_per_episode": 50,
        "save_frequency": 100,
        "eval_frequency": 50,
        "num_eval_episodes": 10,
        "max_grad_norm": 0.5,
    }
