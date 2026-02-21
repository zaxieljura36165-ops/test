"""
系统测试脚本
验证各个模块是否正常工作
"""

import os
import sys
import numpy as np
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.system_config import SystemConfig
from configs.mappo_config import MAPPOConfig
from src.models.network_model import NetworkModel
from src.models.communication_model import CommunicationModel
from src.models.task_model import TaskManager, Task
from src.models.queue_manager import GlobalQueueManager
from src.models.priority_framework import PriorityFramework
from src.models.delay_energy_calculator import DelayEnergyCalculator
from src.models.problem_formulation import OptimizationProblem
from src.algorithms.hierarchical_mappo import HierarchicalMAPPO
from src.environments.simulation_env import SimulationEnvironment

def test_network_model():
    """测试网络模型"""
    print("测试网络模型...")
    config = SystemConfig()
    network = NetworkModel(config)
    
    # 测试基本功能
    assert network.num_vehicles == config.NUM_VEHICLES
    assert network.num_edge_nodes == config.NUM_RSU + config.NUM_MBS
    
    # 测试信道增益计算
    channel_gain = network.get_channel_gain(0, 0)
    assert channel_gain > 0
    
    # 测试数据速率计算
    data_rate = network.calculate_data_rate(0, 0, 0.5)
    assert data_rate >= 0
    
    print("✓ 网络模型测试通过")

def test_communication_model():
    """测试通信模型"""
    print("测试通信模型...")
    config = SystemConfig()
    network = NetworkModel(config)
    comm = CommunicationModel(config, network)
    
    # 测试V2V信道增益
    v2v_gain = comm.get_v2v_channel_gain(0, 1)
    assert v2v_gain >= 0
    
    # 测试可用邻车
    neighbors = comm.get_available_neighbors(0)
    assert isinstance(neighbors, list)
    
    # 测试数据速率计算
    v2i_rate = comm.calculate_v2i_data_rate(0, 0, 0.5)
    v2v_rate = comm.calculate_v2v_data_rate(0, 1, 0.5)
    assert v2i_rate >= 0
    assert v2v_rate >= 0
    
    print("✓ 通信模型测试通过")

def test_task_model():
    """测试任务模型"""
    print("测试任务模型...")
    config = SystemConfig()
    task_manager = TaskManager(config)
    
    # 测试任务生成
    tasks = task_manager.generate_tasks_for_all_vehicles(0)
    assert isinstance(tasks, dict)
    
    # 测试任务创建
    task = Task(
        task_id=0,
        vehicle_id=0,
        data_size=1e6,
        cpu_cycles=1e6,
        deadline=1.0,
        arrival_time=0.0
    )
    
    # 测试任务划分
    task.partition_task(0.5)
    assert task.alpha == 0.5
    assert task.local_data_size == 0.5e6
    assert task.offload_data_size == 0.5e6
    
    print("✓ 任务模型测试通过")

def test_queue_manager():
    """测试队列管理器"""
    print("测试队列管理器...")
    config = SystemConfig()
    queue_manager = GlobalQueueManager(config)
    
    # 测试队列状态
    status = queue_manager.get_global_queue_status()
    assert 'vehicles' in status
    assert 'edge_nodes' in status
    
    # 测试任务添加
    task = Task(0, 0, 1e6, 1e6, 1.0, 0.0)
    queue_manager.add_local_task(0, task, 0.0)
    
    vehicle_status = queue_manager.vehicle_managers[0].get_queue_status()
    assert vehicle_status['hpc_length'] >= 0 or vehicle_status['lpc_length'] >= 0
    
    print("✓ 队列管理器测试通过")

def test_priority_framework():
    """测试优先级框架"""
    print("测试优先级框架...")
    config = SystemConfig()
    network = NetworkModel(config)
    comm = CommunicationModel(config, network)
    priority_framework = PriorityFramework(config)
    
    # 测试优先级计算
    task = Task(0, 0, 1e6, 1e6, 1.0, 0.0)
    priority = priority_framework.priority_calculator.calculate_priority(
        task, 0.0, network, comm
    )
    assert 0 <= priority <= 1
    
    print("✓ 优先级框架测试通过")

def test_delay_energy_calculator():
    """测试时延能耗计算器"""
    print("测试时延能耗计算器...")
    config = SystemConfig()
    network = NetworkModel(config)
    comm = CommunicationModel(config, network)
    queue_manager = GlobalQueueManager(config)
    task_manager = TaskManager(config)
    
    calc = DelayEnergyCalculator(config, network, comm, queue_manager)
    
    # 测试任务创建
    task = Task(0, 0, 1e6, 1e6, 1.0, 0.0)
    task.partition_task(0.5)
    task.local_freq = 1e9
    
    # 测试时延计算
    delay, energy = calc.calculate_local_delay(task, 0.0)
    assert delay >= 0
    assert energy >= 0
    
    # 测试成本计算
    cost = calc.calculate_task_cost(task, 0.0)
    assert cost >= 0
    
    print("✓ 时延能耗计算器测试通过")

def test_optimization_problem():
    """测试优化问题"""
    print("测试优化问题...")
    config = SystemConfig()
    network = NetworkModel(config)
    comm = CommunicationModel(config, network)
    queue_manager = GlobalQueueManager(config)
    task_manager = TaskManager(config)
    
    opt_problem = OptimizationProblem(config, network, comm, queue_manager, task_manager)
    
    # 测试决策变量设置
    opt_problem.set_decision_variables(0, 0, 0.5, 'V2I', 0, 0.5, 1e9)
    
    # 测试约束检查
    constraints = opt_problem.check_constraints()
    assert isinstance(constraints, dict)
    
    # 测试系统状态
    state = opt_problem.get_system_state(0)
    assert isinstance(state, dict)
    
    print("✓ 优化问题测试通过")

def test_hierarchical_mappo():
    """测试分层MAPPO"""
    print("测试分层MAPPO...")
    system_config = SystemConfig()
    mappo_config = MAPPOConfig()
    
    # 强制使用CPU进行测试
    mappo_config.DEVICE = 'cpu'
    
    agent = HierarchicalMAPPO(mappo_config, system_config)
    
    # 测试网络创建
    assert agent.network is not None
    assert agent.optimizers is not None
    
    # 测试动作选择
    state = torch.randn(mappo_config.HIGH_LEVEL_CONFIG['state_dim'], device=agent.device)
    action = agent.select_action(state, 0, None)
    assert isinstance(action, dict)
    
    print("✓ 分层MAPPO测试通过")

def test_simulation_environment():
    """测试仿真环境"""
    print("测试仿真环境...")
    system_config = SystemConfig()
    mappo_config = MAPPOConfig()
    
    env = SimulationEnvironment(system_config, mappo_config)
    
    # 测试环境重置
    initial_state = env.reset()
    assert isinstance(initial_state, dict)
    
    # 测试环境步进
    actions = {0: {'alpha': 0.5, 'mode': 'local', 'freq': 0.5}}
    next_state, reward, done, info = env.step(actions)
    assert isinstance(next_state, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    
    print("✓ 仿真环境测试通过")

def main():
    """主测试函数"""
    print("开始系统测试...")
    print("=" * 50)
    
    try:
        test_network_model()
        test_communication_model()
        test_task_model()
        test_queue_manager()
        test_priority_framework()
        test_delay_energy_calculator()
        test_optimization_problem()
        test_hierarchical_mappo()
        test_simulation_environment()
        
        print("=" * 50)
        print("✓ 所有测试通过！系统运行正常。")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
