# Import các thư viện cần thiết cho Double Deep Q-Network
import numpy as np              # Thư viện tính toán số học
import torch                    # Framework deep learning PyTorch
import torch.nn as nn          # Module neural network của PyTorch
import random                   # Thư viện random cho exploration
from collections import deque   # Cấu trúc dữ liệu deque cho memory buffer
import torch.optim as optim     # Các optimizer cho training

class DDQN(nn.Module):
    """
    Double Deep Q-Network: Mạng neural network cho DDQN
    
    Kiến trúc network giống hệt DQN, khác biệt nằm ở cách sử dụng:
    - DQN: 1 network vừa chọn vừa đánh giá action
    - DDQN: Tách riêng việc chọn (main) và đánh giá (target) action
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Khởi tạo kiến trúc mạng neural network
        
        Args:
            input_size: Kích thước input (số features của state)
            hidden_size: Số neurons trong hidden layers
            output_size: Kích thước output (số actions có thể thực hiện)
        """
        super(DDQN, self).__init__()
        # Tạo mạng neural network với 3 layers fully connected
        self.network = nn.Sequential(
            # Layer 1: Input -> Hidden với ReLU activation
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            # Layer 2: Hidden -> Hidden với ReLU activation
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # Layer 3: Hidden -> Output (Q-values cho mỗi action)
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        """
        Forward pass: Tính toán Q-values cho tất cả actions từ state input
        
        Args:
            x: State tensor (batch_size, input_size)
            
        Returns:
            Q-values tensor (batch_size, output_size)
        """
        return self.network(x)


class DDQNAgent:
    """
    Double Deep Q-Network (DDQN) Agent: Cải tiến của DQN
    
    Khác biệt chính với DQN:
    - DQN: target = r + γ * max_a Q_target(s', a)
    - DDQN: target = r + γ * Q_target(s', argmax_a Q_main(s', a))
    
    Lợi ích của DDQN:
    1. Giảm overestimation bias của Q-values
    2. Tăng stability và accuracy trong training
    3. Cải thiện performance trong nhiều environments
    4. Giảm variance trong Q-value estimates
    
    Nguyên lý hoạt động:
    - Sử dụng main network để CHỌN action tốt nhất
    - Sử dụng target network để ĐÁNH GIÁ Q-value của action đó
    - Tách biệt selection và evaluation để giảm bias
    """
    
    def __init__(self, state_size, action_size, lr=0.001):
        """
        Khởi tạo DDQN Agent với các tham số cần thiết
        
        Args:
            state_size: Kích thước state space (trong RAG system = 5 features)
            action_size: Số lượng actions có thể thực hiện (5 actions: quote, summarize, explain, clarify, fallback)
            lr: Learning rate cho optimizer
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Experience Replay Buffer: Lưu trữ (state, action, reward, next_state, done)
        # Giúp break correlation giữa các experiences liên tiếp và tăng sample efficiency
        self.memory = deque(maxlen=2000)
        
        # Epsilon-Greedy Parameters cho exploration-exploitation tradeoff
        self.epsilon = 1.0          # Bắt đầu với exploration hoàn toàn (100%)
        self.epsilon_min = 0.01     # Minimum exploration rate (1%)
        self.epsilon_decay = 0.995  # Tốc độ giảm epsilon sau mỗi episode
        
        # Tạo 2 neural networks giống hệt nhau
        # Main Q-Network: Được update liên tục, dùng để chọn actions
        self.q_network = DDQN(state_size, 64, action_size)
        # Target Network: Được update định kỳ, dùng để đánh giá Q-values
        self.target_network = DDQN(state_size, 64, action_size)
        
        # Adam optimizer cho việc update weights của main network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Khởi tạo target network với weights giống main network
        self.update_target_network()
        
        # Statistics tracking (optional)
        self.loss_history = []
        self.q_values_history = []
    
    def state_to_tensor(self, state):
        """
        Chuyển đổi state từ encoded integer thành tensor features
        
        Trong RAG system, state được encode thành 1 số integer từ 5 features:
        - Feature 0: Similarity (độ tương tự giữa query và documents)
        - Feature 1: Confidence (độ tin cậy của retrieval)
        - Feature 2: Context (mức độ ngữ cảnh có sẵn)
        - Feature 3: Ambiguity (độ mơ hồ của query)
        - Feature 4: Scope (phạm vi của query)
        
        Mỗi feature có giá trị 0, 1, hoặc 2 (thấp, trung bình, cao)
        State được encode: state = sum(feature[i] * 3^i) với i từ 0 đến 4
        
        Args:
            state: Integer đại diện cho encoded state (0-242)
            
        Returns:
            torch.FloatTensor: Tensor chứa 5 features (1, 5)
        """
        # Decode state từ base-3 system để lấy 5 features
        features = []
        temp_state = state
        for i in range(5):
            # Lấy digit cuối cùng trong hệ base-3
            features.append(temp_state % 3)
            # Loại bỏ digit cuối cùng
            temp_state //= 3
        
        # Chuyển thành tensor và thêm batch dimension
        return torch.FloatTensor(features).unsqueeze(0)
    
    def remember(self, state, action, reward, next_state, done):
        """
        Lưu trữ experience vào replay buffer
        
        Experience Replay là một kỹ thuật quan trọng trong DDQN:
        - Lưu trữ các transitions (s, a, r, s', done) trong circular buffer
        - Sample ngẫu nhiên để training, giúp break temporal correlations
        - Tăng data efficiency bằng cách reuse experiences
        - Stabilize training bằng cách giảm correlation giữa updates
        
        Args:
            state: State hiện tại
            action: Action đã thực hiện (0-4 tương ứng với 5 actions)
            reward: Reward nhận được từ environment
            next_state: State tiếp theo sau khi thực hiện action
            done: Boolean flag, True nếu episode kết thúc
        """
        # Thêm experience tuple vào memory buffer
        # Deque tự động loại bỏ experiences cũ nhất khi đạt maxlen
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        Chọn action dựa trên epsilon-greedy policy
        
        Epsilon-Greedy Strategy cân bằng exploration vs exploitation:
        - Với xác suất epsilon: Random action (exploration)
        - Với xác suất (1-epsilon): Greedy action based on Q-values (exploitation)
        
        Exploration quan trọng vì:
        - Giúp agent khám phá các actions mới
        - Tránh stuck ở local optimum
        - Đảm bảo agent có thể học được optimal policy
        
        Args:
            state: State hiện tại (encoded integer)
            
        Returns:
            int: Action được chọn (0-4)
                0: Quote directly (trích dẫn trực tiếp)
                1: Summarize (tóm tắt)
                2: Explain (diễn giải)
                3: Clarify (hỏi lại để làm rõ)
                4: Fallback (thừa nhận không biết)
        """
        # EXPLORATION: Chọn action ngẫu nhiên với xác suất epsilon
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # EXPLOITATION: Chọn action tốt nhất theo Q-network
        # Chuyển state thành tensor
        state_tensor = self.state_to_tensor(state)
        
        # Forward pass để tính Q-values cho tất cả actions
        with torch.no_grad():  # Không cần gradient khi inference
            q_values = self.q_network(state_tensor)
        
        # Chọn action có Q-value cao nhất
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        """
        Training method cho DDQN: Core của algorithm
        
        DDQN Algorithm Steps:
        1. Sample random batch từ experience replay buffer
        2. Tính current Q-values từ main network
        3. Sử dụng main network để SELECT best actions cho next states
        4. Sử dụng target network để EVALUATE Q-values của actions đó
        5. Compute target Q-values sử dụng Bellman equation
        6. Minimize loss giữa current và target Q-values
        7. Update main network weights
        8. Decay epsilon để giảm exploration theo thời gian
        
        Key Innovation của DDQN:
        - Decoupling action selection từ action evaluation
        - Giảm overestimation bias so với DQN
        
        Args:
            batch_size: Số lượng experiences để sample cho training
        """
        # Kiểm tra có đủ experiences để training không
        if len(self.memory) < batch_size:
            return
        
        # STEP 1: Sample random batch từ memory buffer
        # Random sampling giúp break correlation giữa consecutive experiences
        batch = random.sample(self.memory, batch_size)
        
        # STEP 2: Chuẩn bị data cho training
        # Tách các components từ experience tuples và convert thành tensors
        states = torch.cat([self.state_to_tensor(e[0]) for e in batch])      # Current states
        actions = torch.LongTensor([e[1] for e in batch])                    # Actions taken
        rewards = torch.FloatTensor([e[2] for e in batch])                   # Rewards received
        next_states = torch.cat([self.state_to_tensor(e[3]) for e in batch]) # Next states
        dones = torch.BoolTensor([e[4] for e in batch])                      # Terminal flags
        
        # STEP 3: Tính current Q-values từ main network
        # Lấy Q-value của action đã thực hiện trong từng state
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # STEP 4: DDQN CORE INNOVATION - Double Q-Learning
        # 4a. Sử dụng MAIN network để CHỌN best actions cho next_states
        with torch.no_grad():
            next_q_values_main = self.q_network(next_states)
            next_actions = next_q_values_main.max(1)[1]  # argmax actions
        
        # 4b. Sử dụng TARGET network để ĐÁNH GIÁ Q-values của các actions đó
        with torch.no_grad():
            next_q_values_target = self.target_network(next_states)
            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze()
        
        # STEP 5: Áp dụng Bellman Equation với DDQN modification
        # Q_target = reward + gamma * Q_target(s', argmax_a Q_main(s', a))
        # Nếu episode kết thúc (done=True), chỉ lấy reward
        gamma = 0.95  # Discount factor
        target_q_values = rewards + (gamma * next_q_values * ~dones)
        
        # STEP 6: Tính loss và backpropagation
        # Sử dụng Huber Loss (smooth L1) thay vì MSE để robust hơn với outliers
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        # Lưu loss để tracking (optional)
        self.loss_history.append(loss.item())
        
        # Gradient descent
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()             # Compute gradients
        
        # Gradient clipping để tránh exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()       # Update weights
        
        # STEP 7: Decay epsilon để giảm exploration theo thời gian
        # Bắt đầu với exploration cao, dần dần chuyển sang exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """
        Cập nhật target network với weights từ main network
        
        Target Network Technique trong DDQN:
        - Sử dụng 2 networks với kiến trúc giống hệt nhau
        - Main network: được update liên tục qua gradient descent
        - Target network: được update định kỳ (copy từ main network)
        
        Lý do sử dụng Target Network:
        1. Stability: Targets không thay đổi quá nhanh
        2. Convergence: Giúp algorithm converge tốt hơn
        3. Bias Reduction: Giảm correlation giữa current và target estimates
        
        Frequency: Thường update sau mỗi C steps hoặc episodes
        (trong practice, C = 100-1000 steps tùy vào problem)
        """
        # Hard update: Copy toàn bộ weights từ main sang target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        print(f"Target network updated. Current epsilon: {self.epsilon:.4f}")
    
    def soft_update_target_network(self, tau=0.001):
        """
        Soft update target network (alternative cho hard update)
        
        Thay vì copy hoàn toàn, soft update mixing một phần nhỏ:
        θ_target = τ * θ_main + (1 - τ) * θ_target
        
        Args:
            tau: Mixing coefficient (thường rất nhỏ, 0.001-0.01)
        """
        for target_param, main_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)
    
    def save_model(self, filepath):
        """
        Lưu trained model
        
        Args:
            filepath: Đường dẫn để lưu model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'loss_history': self.loss_history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model
        
        Args:
            filepath: Đường dẫn model đã lưu
        """
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.loss_history = checkpoint.get('loss_history', [])
        print(f"Model loaded from {filepath}")
    
    def get_statistics(self):
        """
        Lấy thống kê training
        
        Returns:
            dict: Dictionary chứa các thống kê
        """
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'avg_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            'total_training_steps': len(self.loss_history)
        }


# Utility functions để sử dụng DDQN trong RAG system
def create_rag_ddqn_agent():
    """
    Tạo DDQN agent cho RAG system với parameters phù hợp
    
    Returns:
        DDQNAgent: Configured agent cho RAG task
    """
    return DDQNAgent(
        state_size=5,      # 5 features: similarity, confidence, context, ambiguity, scope
        action_size=5,     # 5 actions: quote, summarize, explain, clarify, fallback
        lr=0.001          # Learning rate phù hợp cho RAG task
    )

def compare_dqn_vs_ddqn():
    """
    Function để so sánh DQN vs DDQN trong RAG context
    Có thể sử dụng để benchmarking
    """
    print("=== Comparison: DQN vs DDQN ===")
    print("DQN Issues:")
    print("- Overestimation bias in Q-values")
    print("- Less stable training")
    print("- May converge to suboptimal policy")
    print()
    print("DDQN Advantages:")
    print("- Reduced overestimation bias")
    print("- More stable training")
    print("- Better performance in many environments")
    print("- Decoupled action selection and evaluation")
    print()
    print("Use DDQN when:")
    print("- Environment has high variance in rewards")
    print("- Need more stable training")
    print("- Want to reduce overoptimistic estimates")