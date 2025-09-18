# Q-Learning.py 详解

本文档详细解释 `q_learning.py` 文件的功能和实现，这是项目中的off-policy强化学习算法实现。

## 文件概述

`q_learning.py` 实现了Q-learning算法用于疫情控制决策。Q-learning是一种off-policy时序差分学习方法，其核心特点是使用两个不同的策略：行为策略（behavior policy）用于收集经验，目标策略（target policy）用于更新Q值。

## Off-Policy 核心概念

### 策略分离的重要性

**传统混淆的实现：**
```python
# 错误示例：策略概念不清晰
def choose_action(self, state):
    if random() < epsilon:
        return random_action()  # 这是什么策略？
    return argmax(Q[state])     # 这又是什么策略？

def learn(self, state, action, reward, next_state):
    target = reward + gamma * max(Q[next_state])  # 这里用的是哪个策略？
```

**本实现的明确区分：**
```python
# 明确的策略分离
def behavior_policy(self, state):     # 行为策略：用于收集数据
    if random() < epsilon:
        return random_action()
    return argmax(Q[state])

def target_policy(self, state):       # 目标策略：用于评估状态
    return argmax(Q[state])           # 纯贪婪策略

def get_target_value(self, state):    # 目标策略的状态价值
    return max(Q[state])
```

## 主要类：QLearningAgent

### 类初始化

```python
def __init__(self,
             state_size: int,
             action_size: int,
             learning_rate: float = 0.1,
             discount_factor: float = 0.95,
             epsilon: float = 0.1,  # 固定epsilon - 不衰减
             state_bins: int = 8,
             seed: int = None):     # 随机数种子（新增）
```

**关键设计决策：**

1. **固定epsilon**：
   - **传统做法**：epsilon从1.0衰减到0.01
   - **本实现**：epsilon固定为0.1
   - **优势**：更清晰地展示off-policy特性，避免衰减复杂性

2. **状态离散化**：
   - **方法**：将连续状态[0,1]映射到离散bins
   - **总状态数**：8³ = 512个状态
   - **权衡**：平衡表示精度和学习效率

3. **Q表结构**：
   - **维度**：[8, 8, 8, 3] = 512 × 3
   - **索引**：Q[s_bin, i_bin, r_bin, action]
   - **存储**：每个状态-动作对的价值估计

### 状态离散化机制

```python
def _discretize_state(self, state: np.ndarray) -> tuple:
    discretized = []
    for s in state:
        # 将[0,1]映射到[0, state_bins-1]
        bin_idx = min(int(s * self.state_bins), self.state_bins - 1)
        discretized.append(bin_idx)
    return tuple(discretized)
```

**离散化过程示例：**
```python
# 输入状态：[0.75, 0.15, 0.10] (S=75%, I=15%, R=10%)
# state_bins = 8

# S: 0.75 × 8 = 6.0 → bin_idx = 6
# I: 0.15 × 8 = 1.2 → bin_idx = 1  
# R: 0.10 × 8 = 0.8 → bin_idx = 0

# 离散状态：(6, 1, 0)
```

**边界处理：**
- `min(..., state_bins - 1)` 防止索引超出范围
- 例如：1.0 × 8 = 8 → min(8, 7) = 7

### 行为策略 (Behavior Policy)

```python
def behavior_policy(self, state: np.ndarray) -> int:
    """
    行为策略：ε-贪婪策略，用于环境交互时的动作选择
    平衡探索（随机动作）和利用（贪婪动作）
    """
    discrete_state = self._discretize_state(state)
    
    if np.random.random() <= self.epsilon:
        # 探索：随机选择动作
        return np.random.choice(self.action_size)
    else:
        # 利用：选择当前认为最优的动作
        return np.argmax(self.q_table[discrete_state])
```

**行为策略特点：**
1. **探索性**：有ε概率选择随机动作
2. **自适应**：大部分时间选择当前最优动作
3. **数据收集**：为学习算法提供多样化的经验

**概率分布：**
```python
# 对于任意状态s，行为策略的动作概率分布：
# P(a|s) = ε/|A| + (1-ε) * I(a = argmax Q(s,a))
# 其中 I(·) 是指示函数

# 示例：ε=0.1, 动作空间={0,1,2}, 最优动作=1
行为策略概率 = {
    动作0: 0.1/3 = 0.033,      # 仅探索概率
    动作1: 0.1/3 + 0.9 = 0.933, # 探索概率 + 利用概率  
    动作2: 0.1/3 = 0.033       # 仅探索概率
}
```

### 目标策略 (Target Policy)

```python
def target_policy(self, state: np.ndarray) -> int:
    """
    目标策略：贪婪策略，用于Q值更新
    总是选择具有最高Q值的动作
    """
    discrete_state = self._discretize_state(state)
    return np.argmax(self.q_table[discrete_state])

def get_target_value(self, state: np.ndarray) -> float:
    """
    获取目标策略下的状态价值
    用于Q-learning更新中的 max Q(s', a)
    """
    discrete_state = self._discretize_state(state)
    return np.max(self.q_table[discrete_state])
```

**目标策略特点：**
1. **确定性**：总是选择Q值最高的动作
2. **优化导向**：代表我们希望学习的最优策略
3. **评估作用**：用于评估状态价值，而非数据收集

**与行为策略的关系：**
```python
# 在Q值收敛后：
# - 行为策略：90%概率选择最优动作，10%概率随机探索
# - 目标策略：100%概率选择最优动作

# 这种差异使得Q-learning能够：
# 1. 通过行为策略收集多样化经验
# 2. 通过目标策略学习最优价值函数
```

### Q-Learning 更新规则

```python
def learn(self, state: np.ndarray, action: int, reward: float, 
          next_state: np.ndarray, done: bool):
    """
    Q-learning更新规则（off-policy）
    
    核心公式：Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
                                           ^^^^^^^^^^^^
                                           目标策略价值
    """
    discrete_state = self._discretize_state(state)
    
    # 当前Q值
    current_q = self.q_table[discrete_state][action]
    
    if done:
        # 终止状态：没有未来奖励
        target_q = reward
    else:
        # 使用目标策略评估下一状态（off-policy的核心）
        target_q = reward + self.discount_factor * self.get_target_value(next_state)
    
    # Q值更新
    td_error = target_q - current_q
    self.q_table[discrete_state][action] += self.learning_rate * td_error
```

**更新过程详解：**

1. **时序差分目标**：
   ```python
   target_q = reward + γ * max Q(s', a')
   #                         ^^^^^^^^^^^^^
   #                         目标策略评估下一状态
   ```

2. **TD误差计算**：
   ```python
   td_error = target_q - current_q
   #        = [r + γ max Q(s',a')] - Q(s,a)
   ```

3. **Q值更新**：
   ```python
   Q(s,a) ← Q(s,a) + α * td_error
   #      = Q(s,a) + α * [target_q - Q(s,a)]
   ```

**Off-Policy的体现：**
- **数据来源**：动作来自行为策略（ε-贪婪）
- **更新方向**：朝着目标策略（贪婪）的方向更新
- **学习独立性**：Q值更新不依赖于产生数据的策略

### 数值示例

假设当前状态下的Q值为：
```python
Q[state] = [Q(s,0)=-10, Q(s,1)=-5, Q(s,2)=-15]
```

**行为策略动作选择（ε=0.1）：**
```python
最优动作 = argmax([−10, −5, −15]) = 1
行为策略概率 = [0.033, 0.933, 0.033]

# 90%概率选择动作1，10%概率随机探索
```

**假设选择了动作0（探索）：**
```python
状态转移：s → s'
即时奖励：r = -8
下一状态Q值：Q[s'] = [-12, -3, -18]
```

**Q-learning更新：**
```python
# 当前Q值
current_q = Q[s, 0] = -10

# 目标策略评估下一状态
target_value = max(Q[s']) = max([-12, -3, -18]) = -3

# TD目标
target_q = r + γ * target_value = -8 + 0.95 * (-3) = -10.85

# TD误差
td_error = target_q - current_q = -10.85 - (-10) = -0.85

# Q值更新
Q[s, 0] ← Q[s, 0] + α * td_error
        = -10 + 0.1 * (-0.85)
        = -10.085
```

**关键观察：**
1. **动作选择**：使用行为策略（选择了非最优动作0）
2. **价值评估**：使用目标策略（评估下一状态用max）
3. **学习方向**：朝着目标策略的价值估计更新

### 策略信息分析

```python
def get_policy_info(self, state: np.ndarray) -> dict:
    """获取详细的策略分析信息"""
    discrete_state = self._discretize_state(state)
    q_values = self.q_table[discrete_state].copy()
    
    # 目标策略动作（贪婪）
    target_action = np.argmax(q_values)
    
    # 行为策略概率分布（ε-贪婪）
    behavior_probs = np.ones(self.action_size) * (self.epsilon / self.action_size)
    behavior_probs[target_action] += (1 - self.epsilon)
    
    return {
        'discrete_state': discrete_state,
        'q_values': q_values,
        'target_action': target_action,
        'behavior_probabilities': behavior_probs,
        'epsilon': self.epsilon
    }
```

**分析信息用途：**
1. **调试**：检查Q值学习是否合理
2. **策略比较**：观察两种策略的差异
3. **收敛监控**：判断学习是否稳定

## 训练函数：train_q_learning

```python
def train_q_learning(episodes: int = 500, max_steps: int = 100):
    """训练Q-learning智能体"""
    
    # 创建环境和智能体
    env = SIREpidemicEnv(population=5000, max_steps=max_steps, seed=seed)
    agent = QLearningAgent(
        state_size=env.state_size,
        action_size=env.action_space_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1,  # 固定epsilon
        state_bins=8,
        seed=seed     # 随机数种子（新增）
    )
```

### 训练循环

```python
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        # 使用行为策略选择动作（ε-贪婪）
        action = agent.behavior_policy(state)
        
        # 执行动作，获取环境反馈
        next_state, reward, done, info = env.step(action)
        
        # 使用off-policy Q-learning更新
        agent.learn(state, action, reward, next_state, done)
        
        # 状态转移
        state = next_state
        total_reward += reward
        
        if done:
            break
```

**训练流程分析：**
1. **数据收集**：行为策略与环境交互
2. **经验学习**：Q-learning更新朝着目标策略
3. **策略改进**：Q值改善使得两个策略都变得更好

### 学习曲线分析

```python
# 典型的Q-learning学习曲线特征：

初期（Episodes 1-100）：
- 奖励波动较大：由于随机探索
- 平均奖励上升：Q值逐渐学习

中期（Episodes 100-300）：  
- 波动减小：策略趋于稳定
- 持续改善：Q值接近最优

后期（Episodes 300-500）：
- 收敛稳定：奖励变化很小
- 偶有波动：ε=0.1的探索影响
```

## 测试函数：test_q_learning

```python
def test_q_learning():
    """测试训练好的Q-learning智能体"""
    
    # 加载训练好的模型
    agent.load_model('models/original/q_learning.pkl')
    
    # 测试时使用目标策略（纯贪婪）
    for step in range(100):
        action = agent.choose_action(state, use_target_policy=True)
        state, reward, done, info = env.step(action)
        
        if done:
            break
```

**测试特点：**
1. **策略选择**：使用目标策略（最优策略）进行测试
2. **无探索**：测试时不进行随机探索
3. **性能评估**：评估学习到的最优策略效果

## 算法优势

### 1. Off-Policy学习的优势

**数据利用效率：**
```python
# Q-learning可以从任何策略收集的数据中学习
collected_data = [
    (s1, a1, r1, s1'),  # 来自随机策略
    (s2, a2, r2, s2'),  # 来自ε-贪婪策略  
    (s3, a3, r3, s3'),  # 来自专家策略
]

# 所有数据都能用于改进目标策略
for transition in collected_data:
    agent.learn(*transition)
```

**探索与学习分离：**
- **行为策略**：负责探索环境，收集多样化经验
- **目标策略**：专注于优化，不受探索噪声影响

### 2. 固定Epsilon的教学价值

**概念清晰：**
```python
# 传统衰减epsilon的问题：
# 初期：ε=1.0 → 完全随机，难以观察策略差异
# 后期：ε=0.01 → 几乎贪婪，off-policy特性不明显

# 固定epsilon的优势：
# 始终：ε=0.1 → 策略差异稳定，便于分析理解
```

**策略对比：**
```python
# 在任意时刻都能清晰对比：
behavior_policy_prob = [0.033, 0.933, 0.033]  # ε-贪婪
target_policy_prob = [0, 1, 0]                # 纯贪婪

# 差异明显，便于理解off-policy机制
```

## 与SARSA的关键区别

| 特性 | Q-Learning (Off-Policy) | SARSA (On-Policy) |
|------|-------------------------|-------------------|
| **策略数量** | 2个（行为策略 ≠ 目标策略） | 1个（单一策略） |
| **动作选择** | behavior_policy() | epsilon_greedy_policy() |
| **Q值更新** | max Q(s',a') | Q(s',a') 实际动作 |
| **学习目标** | 最优策略 | 当前策略 |
| **探索影响** | 不影响学习目标 | 影响学习目标 |

**更新公式对比：**
```python
# Q-learning：
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
#                           ^^^^^^^^^^^^
#                           与实际选择无关

# SARSA：  
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
#                           ^^^^^^^
#                           实际选择的动作
```

## 使用示例

### 基本训练
```python
from q_learning import train_q_learning, test_q_learning

# 训练智能体（可复现）
agent, env = train_q_learning(episodes=500, max_steps=100, seed=42)

# 测试智能体
test_q_learning(seed=42)
```

### 完整IRL实验（推荐）
```bash
# 一键运行完整实验：Q-learning训练 + IRL学习 + 可视化对比
python run_irl_experiment.py
```

**自动生成的可视化对比图：**
- `outputs/plots/q_learning_policy_comparison.png`
  - 原始Q-learning策略 vs IRL重训练策略
  - 疫情曲线对比 + 动作序列对比
  - 性能指标改进百分比

### 策略分析
```python
from q_learning import QLearningAgent
from environment import SIREpidemicEnv

# 创建智能体
agent = QLearningAgent(state_size=3, action_size=3, epsilon=0.1, seed=42)

# 分析特定状态
state = np.array([0.7, 0.2, 0.1])  # 疫情中期状态
policy_info = agent.get_policy_info(state)

print("策略分析:")
print(f"Q值: {policy_info['q_values']}")
print(f"目标策略动作: {policy_info['target_action']}")
print(f"行为策略概率: {policy_info['behavior_probabilities']}")
```

### 参数实验
```python
# 比较不同epsilon值的影响
epsilons = [0.0, 0.1, 0.2, 0.3]
results = {}

for eps in epsilons:
    agent = QLearningAgent(epsilon=eps)
    rewards = train_agent(agent, episodes=200)
    results[eps] = np.mean(rewards[-50:])  # 最后50个episode的平均奖励

print("不同epsilon的性能:")
for eps, reward in results.items():
    print(f"ε={eps}: 平均奖励={reward:.1f}")
```

## 总结

`q_learning.py` 实现了一个教学友好的off-policy Q-learning算法，其核心贡献包括：

### 理论贡献
1. **明确的策略分离**：清晰区分行为策略和目标策略
2. **固定epsilon设计**：便于理解off-policy机制
3. **完整的分析工具**：支持策略行为的深入分析

### 实现特点
1. **代码清晰**：每个方法都有明确的概念对应
2. **注释详细**：解释了算法的每个关键步骤
3. **功能完整**：包含训练、测试、分析的完整流程

### 教育价值
1. **概念示范**：展示了off-policy学习的本质
2. **对比基础**：为与SARSA等算法的比较提供基础
3. **扩展平台**：可作为实现更复杂算法的起点

该实现不仅解决了疫情控制的实际问题，更重要的是提供了理解强化学习核心概念的清晰范例，特别是off-policy学习中行为策略与目标策略分离的重要思想。