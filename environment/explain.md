# Hướng dẫn Setup Môi trường RL cho RAG Agent

## 1. Tổng quan về Môi trường MDP

### Định nghĩa MDP (Markov Decision Process)
```
- States (S): 3^5 = 243 trạng thái có thể
- Actions (A): 5 hành động khác nhau  
- Rewards (R): Phần thưởng từ -12 đến +3
- Discount Factor (γ): 0.95
```

### Không gian trạng thái
Mỗi trạng thái được mã hóa từ 5 đặc trưng, mỗi đặc trưng có 3 mức độ (0,1,2):
- **Similarity**: Độ tương đồng ngữ nghĩa query-documents
- **Confidence**: Độ tin cậy từ rerank score  
- **Context**: Mức độ phù hợp ngữ cảnh hội thoại
- **Ambiguity**: Độ mơ hồ của câu hỏi
- **Scope**: Mức độ khớp phạm vi query-documents

## 2. Chi tiết các Components

### 2.1 State Encoding/Decoding
```python
# Mã hóa: features [1,2,0,1,2] → state = 1×3⁰ + 2×3¹ + 0×3² + 1×3³ + 2×3⁴
state = Σ(feature[i] × 3^i)

# Giải mã: state → features bằng phép chia lấy dư
for i in range(5):
    features[i] = state % 3
    state //= 3
```

### 2.2 Feature Extraction Pipeline
```
Query + Retrieved Docs → Feature Extractor → [s1,s2,s3,s4,s5] → State ID
```

**Similarity (s1)**: Cosine similarity trung bình
- 2: avg_score > 0.8 (cao)
- 1: 0.5 < avg_score ≤ 0.8 (trung bình)  
- 0: avg_score ≤ 0.5 (thấp)

**Confidence (s2)**: Rerank score tối đa
- 2: max_score > 0.9 (tin cậy cao)
- 1: 0.6 < max_score ≤ 0.9 (tin cậy vừa)
- 0: max_score ≤ 0.6 (tin cậy thấp)

**Context (s3)**: Ngữ cảnh hội thoại (hiện tại random)
**Ambiguity (s4)**: Đếm từ khóa mơ hồ
**Scope (s5)**: Embedding similarity trung bình

## 3. Action Space

| Action ID | Mô tả | Khi nào sử dụng |
|-----------|-------|-----------------|
| 0 | Trích dẫn nguyên văn | Similarity cao + Confidence cao |
| 1 | Tóm tắt nội dung | Confidence ≥ 1 + Context ≥ 1 |
| 2 | Diễn giải lại | Ambiguity ≥ 1 + Scope ≥ 1 |
| 3 | Hỏi lại để làm rõ | Ambiguity cao (đặc biệt = 2) |
| 4 | Thoái lui an toàn | Confidence thấp + Similarity thấp |

## 4. Reward Function Design

### Nguyên tắc thiết kế phần thưởng:
- **Positive rewards**: Khuyến khích hành động đúng ngữ cảnh
- **Negative rewards**: Phạt hành động không phù hợp
- **Magnitude scaling**: Phạt nặng hành động có rủi ro cao

### Ví dụ reward structure:
```python
Action 0 (Trích dẫn):
  +3: similarity=2 AND confidence=2  # Tối ưu
  +2: similarity≥1 AND confidence≥1  # Tốt
  -1: Các trường hợp khác           # Không phù hợp

Action 4 (Thoái lui):
  +1: confidence=0 AND similarity=0  # Đúng khi không chắc chắn
  -12: Các trường hợp khác          # Phạt nặng thoái lui không cần thiết
```

## 5. Setup Môi trường Training

### 5.1 Dependencies
```python
# Core libraries
import numpy as np
import random
from helper.helper_functions import embedding_function, rerank_function

# RL libraries (cần thêm)
import torch
import torch.nn as nn
from stable_baselines3 import DQN, PPO
```

### 5.2 Environment Interface
```python
class RAGEnvironment:
    def __init__(self):
        # MDP parameters
        self.n_states = 3**5
        self.n_actions = 5
        self.gamma = 0.95
        
    def step(self, action):
        # Trả về: next_state, reward, done, info
        
    def reset(self):
        # Trả về: initial_state
        
    def extract_features(self, query, docs):
        # Trả về: [s1, s2, s3, s4, s5]
```

### 5.3 Training Loop
```python
env = RAGEnvironment()
agent = DQN('MlpPolicy', env, verbose=1)

# Training
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    
    while not done:
        action = agent.predict(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
```

## 6. Những điểm cần cải thiện

### 6.1 State Transition Model
```python
# Hiện tại: chuyển đổi ngẫu nhiên
self.current_state = random.randint(0, self.n_states - 1)

# Cần: mô hình chuyển đổi thực tế
next_features = self.update_features_based_on_action(current_features, action)
```

### 6.2 Context Management
- Thêm lịch sử hội thoại vào state
- Memory mechanism cho long-term context
- Dynamic context window

### 6.3 Reward Shaping
- Thêm intermediate rewards
- Curriculum learning
- Human feedback integration

### 6.4 Evaluation Metrics
```python
metrics = {
    'relevance_score': compute_relevance(response, ground_truth),
    'safety_score': check_safety_violations(response),
    'user_satisfaction': get_user_feedback(),
    'response_time': measure_latency()
}
```

## 7. Implementation Roadmap

### Phase 1: Basic Setup
1. Implement helper functions (embedding_function, rerank_function)
2. Test feature extraction với real data
3. Validate reward function logic

### Phase 2: RL Integration  
1. Integrate với RL library (Stable-Baselines3)
2. Design training data pipeline
3. Implement evaluation framework

### Phase 3: Advanced Features
1. Multi-turn conversation support
2. Dynamic reward adjustment
3. Transfer learning capabilities

### Phase 4: Production
1. A/B testing framework
2. Online learning capabilities
3. Performance monitoring

## 8. Key Challenges & Solutions

**Challenge**: State space quá lớn (243 states)
**Solution**: Function approximation với neural networks

**Challenge**: Sparse rewards
**Solution**: Reward shaping + curriculum learning

**Challenge**: Non-stationary environment
**Solution**: Online adaptation + experience replay

**Challenge**: Safety constraints
**Solution**: Constrained RL + safe exploration strategies