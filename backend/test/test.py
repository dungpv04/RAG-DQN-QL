import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# ========================= MDP Environment =========================
class RAGEnvironment:
    def __init__(self):
        # Định nghĩa MDP
        self.n_states = 3**5  # 243 trạng thái (5 đặc trưng, mỗi đặc trưng 3 mức)
        self.n_actions = 5    # 5 hành động
        self.gamma = 0.95     # Hệ số chiết khấu
        
        # Hành động và phần thưởng
        self.actions = {
            0: "trích dẫn nguyên văn",
            1: "tóm tắt nội dung", 
            2: "diễn giải lại",
            3: "hỏi lại để làm rõ",
            4: "thoái lui an toàn"
        }
        
        self.current_state = 0
        
    def encode_state(self, features):
        """Mã hóa 5 đặc trưng thành trạng thái số"""
        state = 0
        for i, feature in enumerate(features):
            state += feature * (3 ** i)
        return state
    
    def decode_state(self, state):
        """Giải mã trạng thái số thành 5 đặc trưng"""
        features = []
        for i in range(5):
            features.append(state % 3)
            state //= 3
        return features
    
    def extract_features(self, query, retrieved_docs):
        """Trích xuất 5 đặc trưng từ câu hỏi và tài liệu truy xuất"""
        # Giả định các hàm phụ trợ đã tồn tại
        similarity = self._get_similarity_level(query, retrieved_docs)
        confidence = self._get_retrieval_confidence(retrieved_docs)
        context = self._get_context_match(query, retrieved_docs)
        ambiguity = self._get_ambiguity_level(query)
        scope = self._get_scope_match(query, retrieved_docs)
        
        return [similarity, confidence, context, ambiguity, scope]
    
    def _get_similarity_level(self, query, docs):
        """Mức độ tương đồng ngữ nghĩa (0: thấp, 1: trung bình, 2: cao)"""
        # Tính similarity đơn giản dựa trên từ khóa chung
        query_words = set(query.lower().split())
        if not docs:
            return 0
        
        max_similarity = 0
        for doc in docs[:3]:  # Chỉ kiểm tra 3 doc đầu tiên
            doc_words = set(str(doc).lower().split())
            common_words = len(query_words & doc_words)
            similarity = common_words / max(len(query_words), 1)
            max_similarity = max(max_similarity, similarity)
        
        if max_similarity > 0.3: return 2
        elif max_similarity > 0.1: return 1
        else: return 0
    
    def _get_retrieval_confidence(self, docs):
        """Độ tin cậy truy hồi dựa trên độ dài và chất lượng docs"""
        if not docs: return 0
        
        # Tính confidence dựa trên độ dài và số lượng docs
        total_length = sum(len(str(doc).split()) for doc in docs[:3])
        if total_length > 100: return 2
        elif total_length > 30: return 1
        else: return 0
    
    def _get_context_match(self, query, docs):
        """Ngữ cảnh hội thoại phù hợp"""
        # Logic đơn giản để demo
        return random.randint(0, 2)
    
    def _get_ambiguity_level(self, query):
        """Độ mơ hồ trong câu hỏi"""
        ambiguous_words = ["có thể", "như thế nào", "gì", "tại sao", "thế nào", "ra sao", "bao nhiêu"]
        question_words = ["có", "là", "được", "phải", "nên", "khi nào", "ở đâu"]
        
        query_lower = query.lower()
        ambiguous_count = sum(1 for word in ambiguous_words if word in query_lower)
        question_count = sum(1 for word in question_words if word in query_lower)
        
        if ambiguous_count >= 2 or (ambiguous_count >= 1 and question_count >= 2): return 2
        elif ambiguous_count >= 1 or question_count >= 1: return 1
        else: return 0
    
    def _get_scope_match(self, query, docs):
        """Phù hợp phạm vi - kiểm tra từ khóa chuyên ngành"""
        education_keywords = ["học phí", "tín chỉ", "học kỳ", "tốt nghiệp", "điểm", "môn học", 
                            "sinh viên", "trường", "đại học", "kỷ luật", "thôi học", "đăng ký"]
        
        query_lower = query.lower()
        keyword_count = sum(1 for keyword in education_keywords if keyword in query_lower)
        
        if keyword_count >= 2: return 2
        elif keyword_count >= 1: return 1
        else: return 0
    
    def get_reward(self, state, action):
        """Hàm phần thưởng dựa trên trạng thái và hành động"""
        features = self.decode_state(state)
        similarity, confidence, context, ambiguity, scope = features
        
        # Logic phần thưởng phức tạp
        if action == 0:  # Trích dẫn nguyên văn
            if similarity == 2 and confidence == 2: return 3
            elif similarity >= 1 and confidence >= 1: return 2
            else: return -1
            
        elif action == 1:  # Tóm tắt nội dung
            if confidence >= 1 and context >= 1: return 2
            elif confidence == 0: return -5
            else: return 1
            
        elif action == 2:  # Diễn giải lại
            if ambiguity >= 1 and scope >= 1: return 2
            else: return -1
            
        elif action == 3:  # Hỏi lại để làm rõ
            if ambiguity == 2: return 3
            elif ambiguity == 1: return 1
            else: return -8
            
        elif action == 4:  # Thoái lui an toàn
            if confidence == 0 and similarity == 0: return 1
            else: return -12
            
        return 0
    
    def step(self, action):
        """Thực hiện hành động và trả về trạng thái mới, phần thưởng"""
        reward = self.get_reward(self.current_state, action)
        # Giả định chuyển đổi trạng thái ngẫu nhiên cho demo
        self.current_state = random.randint(0, self.n_states - 1)
        done = False  # Trong thực tế, có thể kết thúc episode
        return self.current_state, reward, done
    
    def reset(self):
        """Reset môi trường về trạng thái ban đầu"""
        self.current_state = random.randint(0, self.n_states - 1)
        return self.current_state

# ========================= Q-Learning Algorithm =========================
class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
    def choose_action(self, state):
        """Epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, gamma=0.95):
        """Cập nhật Q-table"""
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        new_value = old_value + self.lr * (reward + gamma * next_max - old_value)
        self.q_table[state, action] = new_value
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ========================= DQN Algorithm =========================
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Neural networks
        self.q_network = DQN(state_size, 64, action_size)
        self.target_network = DQN(state_size, 64, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Copy weights to target network
        self.update_target_network()
    
    def state_to_tensor(self, state):
        """Chuyển đổi trạng thái thành tensor"""
        # Decode state để lấy features
        features = []
        temp_state = state
        for i in range(5):
            features.append(temp_state % 3)
            temp_state //= 3
        return torch.FloatTensor(features).unsqueeze(0)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = self.state_to_tensor(state)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.cat([self.state_to_tensor(e[0]) for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.cat([self.state_to_tensor(e[3]) for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# ========================= Training và So sánh =========================
def train_q_learning(episodes=500):
    """Huấn luyện Q-Learning với dữ liệu thực tế"""
    env = RAGEnvironment()
    agent = QLearningAgent(env.n_states, env.n_actions, lr=0.3, epsilon=1.0)  # Tăng learning rate
    
    # Tạo training data với các query mẫu
    training_queries = [
        "Học phí được tính như thế nào",
        "Thời gian học tối đa là bao lâu", 
        "Điều kiện tốt nghiệp là gì",
        "Khi nào bị buộc thôi học",
        "Cách đăng ký học lại môn trượt",
        "Quy định về điểm số",
        "Học kỳ phụ có được tổ chức không",
        "Số tín chỉ tối thiểu mỗi học kỳ"
    ]
    
    rewards_history = []
    
    for episode in range(episodes):
        total_reward = 0
        
        # Mỗi episode huấn luyện với nhiều query
        for _ in range(10):  # 10 bước mỗi episode
            # Chọn query ngẫu nhiên
            query = random.choice(training_queries)
            
            # Tạo retrieved docs giả lập
            docs = ["Quy chế đào tạo trình độ đại học", "Thời gian học chuẩn 4 năm", "Điểm tối thiểu để tốt nghiệp"]
            
            # Trích xuất features
            features = env.extract_features(query, docs)
            state = env.encode_state(features)
            
            # Chọn action
            action = agent.choose_action(state)
            
            # Tính reward dựa trên độ phù hợp của action
            reward = env.get_reward(state, action)
            
            # Tạo next state
            next_state = random.randint(0, env.n_states - 1)
            
            # Update Q-table
            agent.learn(state, action, reward, next_state, env.gamma)
            
            total_reward += reward
        
        rewards_history.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Q-Learning Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, rewards_history

def train_dqn(episodes=500):
    """Huấn luyện DQN với dữ liệu thực tế"""
    env = RAGEnvironment()
    agent = DQNAgent(5, env.n_actions, lr=0.01)  # Tăng learning rate
    
    training_queries = [
        "Học phí được tính như thế nào",
        "Thời gian học tối đa là bao lâu", 
        "Điều kiện tốt nghiệp là gì",
        "Khi nào bị buộc thôi học",
        "Cách đăng ký học lại môn trượt"
    ]
    
    rewards_history = []
    
    for episode in range(episodes):
        total_reward = 0
        
        for _ in range(10):
            query = random.choice(training_queries)
            docs = ["Quy chế đào tạ", "Thời gian học", "Điểm số"]
            
            features = env.extract_features(query, docs)
            state = env.encode_state(features)
            
            action = agent.act(state)
            reward = env.get_reward(state, action)
            
            next_state = random.randint(0, env.n_states - 1)
            done = False
            
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward
        
        agent.replay()
        
        if episode % 50 == 0:
            agent.update_target_network()
        
        rewards_history.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"DQN Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return agent, rewards_history

# ========================= Demo và Đánh giá =========================
def compare_algorithms():
    """So sánh hiệu quả của Q-Learning và DQN"""
    print("=== Bắt đầu huấn luyện Q-Learning ===")
    q_agent, q_rewards = train_q_learning(episodes=300)
    
    print("\n=== Bắt đầu huấn luyện DQN ===")
    dqn_agent, dqn_rewards = train_dqn(episodes=300)
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(q_rewards, label='Q-Learning', alpha=0.7)
    plt.plot(dqn_rewards, label='DQN', alpha=0.7)
    plt.title('Reward theo Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    
    # Tính moving average
    window = 30
    if len(q_rewards) >= window:
        q_ma = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
        dqn_ma = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
        
        plt.subplot(1, 2, 2)
        plt.plot(q_ma, label='Q-Learning (MA)', linewidth=2)
        plt.plot(dqn_ma, label='DQN (MA)', linewidth=2)
        plt.title(f'Moving Average Reward (window={window})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return q_agent, dqn_agent

# ========================= Giả định các hàm phụ trợ =========================
def embedding_function(query, doc):
    """Giả định hàm tính similarity"""
    return random.uniform(0, 1)

def rerank_function(doc):
    """Giả định hàm rerank"""
    return random.uniform(0, 1)

def get_retrieved_text(docs):
    """Giả định hàm lấy text từ docs"""
    return " ".join(docs) if docs else ""

# ========================= Ứng dụng Chatbot =========================
class RAGChatbot:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        
    def respond(self, query):
        """Trả lời câu hỏi của sinh viên"""
        # Giả định truy xuất tài liệu
        retrieved_docs = ["Document 1", "Document 2", "Document 3"]
        
        # Trích xuất đặc trưng
        features = self.env.extract_features(query, retrieved_docs)
        state = self.env.encode_state(features)
        
        # Chọn hành động tốt nhất
        if isinstance(self.agent, QLearningAgent):
            action = np.argmax(self.agent.q_table[state])
        else:  # DQN
            action = self.agent.act(state)
        
        action_name = self.env.actions[action]
        
        # Thực hiện hành động tương ứng
        if action == 0:  # Trích dẫn nguyên văn
            response = f"Theo tài liệu: {get_retrieved_text(retrieved_docs[:1])}"
        elif action == 1:  # Tóm tắt
            response = f"Tóm tắt: {get_retrieved_text(retrieved_docs)[:100]}..."
        elif action == 2:  # Diễn giải
            response = f"Có thể hiểu rằng: {get_retrieved_text(retrieved_docs)[:80]}..."
        elif action == 3:  # Hỏi lại
            response = "Bạn có thể làm rõ thêm câu hỏi được không?"
        else:  # Thoái lui
            response = "Xin lỗi, tôi chưa có đủ thông tin để trả lời câu hỏi này."
        
        return response, action_name

# ========================= Main Execution =========================
if __name__ == "__main__":
    print("=== RAG Chatbot với Reinforcement Learning ===\n")
    
    # So sánh thuật toán
    q_agent, dqn_agent = compare_algorithms()
    
    # Demo chatbot
    env = RAGEnvironment()
    
    print("\n=== Demo Chatbot với Q-Learning ===")
    chatbot_q = RAGChatbot(q_agent, env)
    
    sample_queries = [
        "Học phí của trường là bao nhiêu?",
        "Thời gian học của khóa học?",
        "Yêu cầu tuyển sinh như thế nào?"
    ]
    
    for query in sample_queries:
        response, action = chatbot_q.respond(query)
        print(f"Câu hỏi: {query}")
        print(f"Hành động: {action}")
        print(f"Trả lời: {response}\n")
    
    print("=== Demo Chatbot với DQN ===")
    chatbot_dqn = RAGChatbot(dqn_agent, env)
    
    for query in sample_queries:
        response, action = chatbot_dqn.respond(query)
        print(f"Câu hỏi: {query}")
        print(f"Hành động: {action}")
        print(f"Trả lời: {response}\n")