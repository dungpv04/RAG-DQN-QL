# Import các thư viện cần thiết cho Deep Q-Network
import numpy as np              # Thư viện tính toán số học
import torch                    # Framework deep learning PyTorch
import torch.nn as nn          # Module neural network của PyTorch
import random                   # Thư viện random cho exploration
from collections import deque   # Cấu trúc dữ liệu deque cho memory buffer
import torch.optim as optim     # Các optimizer cho training

class DQN(nn.Module):
    """
    Deep Q-Network: Mạng neural network để xấp xỉ hàm Q-value
    
    Thay vì sử dụng Q-table như Q-Learning truyền thống, DQN sử dụng
    neural network để học mapping từ state -> Q-values cho mỗi action
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Khởi tạo kiến trúc mạng neural network
        
        Args:
            input_size: Kích thước input (số features của state)
            hidden_size: Số neurons trong hidden layers
            output_size: Kích thước output (số actions có thể thực hiện)
        """
        super(DQN, self).__init__()
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

class DQNAgent:
    """
    DQN Agent: Triển khai thuật toán Deep Q-Network
    
    Các thành phần chính:
    1. Experience Replay: Lưu trữ và replay experiences
    2. Target Network: Ổn định quá trình training 
    3. Epsilon-Greedy: Cân bằng exploration vs exploitation
    """
    def __init__(self, state_size, action_size, lr=0.001):
        """
        Khởi tạo DQN Agent với các tham số cần thiết
        
        Args:
            state_size: Kích thước state space
            action_size: Số lượng actions có thể thực hiện
            lr: Learning rate cho optimizer
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Experience Replay Buffer: Lưu trữ (state, action, reward, next_state, done)
        # Giúp break correlation giữa các experiences liên tiếp
        self.memory = deque(maxlen=2000)
        
        # Epsilon-Greedy Parameters cho exploration
        self.epsilon = 1.0          # Bắt đầu với exploration hoàn toàn (100%)
        self.epsilon_min = 0.01     # Minimum exploration rate (1%)
        self.epsilon_decay = 0.995  # Tốc độ giảm epsilon sau mỗi episode
        
        # Tạo 2 neural networks giống hệt nhau
        # Main Q-Network: Được update liên tục trong quá trình training
        self.q_network = DQN(state_size, 64, action_size)
        # Target Network: Được update định kỳ để stabilize training
        self.target_network = DQN(state_size, 64, action_size)
        
        # Adam optimizer cho việc update weights
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Khởi tạo target network với weights giống main network
        self.update_target_network()
    
    def state_to_tensor(self, state):
        """
        Chuyển đổi state từ encoded integer thành tensor features
        
        Trong project này, state được encode thành 1 số integer duy nhất.
        Method này decode lại thành 5 features riêng biệt:
        - Feature 0: Similarity (độ tương tự)
        - Feature 1: Confidence (độ tin cậy)
        - Feature 2: Context (ngữ cảnh)
        - Feature 3: Ambiguity (độ mơ hồ)
        - Feature 4: Scope (phạm vi)
        
        Mỗi feature có giá trị 0, 1, hoặc 2 (thấp, trung bình, cao)
        
        Args:
            state: Integer đại diện cho encoded state
            
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
        
        Experience Replay là một kỹ thuật quan trọng trong DQN:
        - Lưu trữ các transitions (s, a, r, s', done) trong buffer
        - Sau đó sample ngẫu nhiên để training
        - Giúp break correlation giữa các samples liên tiếp
        - Tăng data efficiency và stability
        
        Args:
            state: State hiện tại
            action: Action đã thực hiện
            reward: Reward nhận được
            next_state: State tiếp theo
            done: Có phải là terminal state không
        """
        # Thêm experience tuple vào memory buffer
        # Deque tự động loại bỏ experiences cũ khi đạt maxlen
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        Chọn action dựa trên epsilon-greedy policy
        
        Epsilon-Greedy Strategy:
        - Với xác suất epsilon: Chọn action ngẫu nhiên (exploration)
        - Với xác suất (1-epsilon): Chọn action có Q-value cao nhất (exploitation)
        
        Điều này giúp cân bằng giữa:
        - Exploration: Khám phá actions mới để tìm strategies tốt hơn
        - Exploitation: Sử dụng kiến thức đã học để maximize reward
        
        Args:
            state: State hiện tại
            
        Returns:
            int: Action được chọn (0 đến action_size-1)
        """
        # EXPLORATION: Chọn action ngẫu nhiên với xác suất epsilon
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # EXPLOITATION: Chọn action tốt nhất theo Q-network
        # Chuyển state thành tensor
        state_tensor = self.state_to_tensor(state)
        # Forward pass để tính Q-values cho tất cả actions
        q_values = self.q_network(state_tensor)
        # Chọn action có Q-value cao nhất
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        """
        Training method: Học từ experiences đã lưu trữ
        
        Đây là heart của DQN algorithm:
        1. Sample random batch từ memory buffer
        2. Tính current Q-values từ main network
        3. Tính target Q-values từ target network
        4. Update main network để minimize loss
        
        Args:
            batch_size: Số lượng experiences để sample cho training
        """
        # Kiểm tra có đủ experiences để training không
        if len(self.memory) < batch_size:
            return
        
        # STEP 1: Sample random batch từ memory buffer
        # Điều này giúp break correlation giữa các consecutive experiences
        batch = random.sample(self.memory, batch_size)
        
        # STEP 2: Chuẩn bị data cho training
        # Tách các components từ experiences và chuyển thành tensors
        states = torch.cat([self.state_to_tensor(e[0]) for e in batch])      # Current states
        actions = torch.LongTensor([e[1] for e in batch])                    # Actions taken
        rewards = torch.FloatTensor([e[2] for e in batch])                   # Rewards received
        next_states = torch.cat([self.state_to_tensor(e[3]) for e in batch]) # Next states
        dones = torch.BoolTensor([e[4] for e in batch])                      # Terminal flags
        
        # STEP 3: Tính current Q-values từ main network
        # Lấy Q-value của action đã thực hiện trong từng state
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # STEP 4: Tính target Q-values từ target network
        # Lấy max Q-value của next_state (Bellman equation)
        # detach() để không tính gradient cho target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        
        # STEP 5: Áp dụng Bellman Equation
        # Q_target = reward + gamma * max(Q(next_state)) if not done
        # Q_target = reward if done (terminal state)
        # Gamma = 0.95 là discount factor
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        # STEP 6: Tính loss và backpropagation
        # Sử dụng Mean Squared Error giữa predicted và target Q-values
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Zero gradients, backward pass, và update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # STEP 7: Decay epsilon để giảm exploration theo thời gian
        # Bắt đầu với exploration cao, dần dần chuyển sang exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """
        Cập nhật target network với weights từ main network
        
        Target Network Technique:
        - Sử dụng 2 networks giống hệt nhau
        - Main network được update liên tục
        - Target network được update định kỳ (ít thường xuyên)
        
        Lý do sử dụng Target Network:
        - Trong DQN, ta dùng neural network để estimate Q-values
        - Nếu chỉ dùng 1 network, target sẽ thay đổi liên tục
        - Điều này gây instability trong training
        - Target network giúp stabilize targets trong một khoảng thời gian
        
        Thường được gọi sau mỗi N episodes hoặc N steps
        """
        # Copy tất cả weights và biases từ main network sang target network
        self.target_network.load_state_dict(self.q_network.state_dict())