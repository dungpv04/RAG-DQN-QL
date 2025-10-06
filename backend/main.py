import numpy as np
import matplotlib.pyplot as plt
from chat.rag_chatbot import RAGChatbot
from environment.rag_environment import RAGEnvironment
from algorithm.q_learning import QLearningAgent
from algorithm.dqn import DQNAgent
def train_q_learning(episodes=1000):
    """Huấn luyện Q-Learning"""
    env = RAGEnvironment()
    agent = QLearningAgent(env.n_states, env.n_actions)
    
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(50):  # Giới hạn số bước mỗi episode
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            agent.learn(state, action, reward, next_state, env.gamma)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        rewards_history.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Q-Learning Episode {episode}, Avg Reward: {avg_reward:.2f}")
    
    return agent, rewards_history

def train_dqn(episodes=1000):
    """Huấn luyện DQN"""
    env = RAGEnvironment()
    agent = DQNAgent(5, env.n_actions)  # 5 features as input
    
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(50):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        agent.replay()
        
        if episode % 100 == 0:
            agent.update_target_network()
        
        rewards_history.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"DQN Episode {episode}, Avg Reward: {avg_reward:.2f}")
    
    return agent, rewards_history

# ========================= Demo và Đánh giá =========================
def compare_algorithms():
    """So sánh hiệu quả của Q-Learning và DQN"""
    print("=== Bắt đầu huấn luyện Q-Learning ===")
    q_agent, q_rewards = train_q_learning(episodes=500)
    
    print("\n=== Bắt đầu huấn luyện DQN ===")
    dqn_agent, dqn_rewards = train_dqn(episodes=500)
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(q_rewards, label='Q-Learning', alpha=0.7)
    plt.plot(dqn_rewards, label='DQN', alpha=0.7)
    plt.title('Reward theo Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    
    # Tính moving average
    window = 50
    q_ma = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
    dqn_ma = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
    
    plt.subplot(1, 2, 2)
    plt.plot(q_ma, label='Q-Learning (MA)', linewidth=2)
    plt.plot(dqn_ma, label='DQN (MA)', linewidth=2)
    plt.title(f'Moving Average Reward (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return q_agent, dqn_agent

# ========================= Giả định các hàm phụ trợ =========================

# ========================= Ứng dụng Chatbot =========================


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
        "Quy chế đào tạo đại học quy định những gì?",
        "Thời gian học tập tối đa cho ngành Kinh tế hệ chính quy là bao lâu?",
        "Định nghĩa của học phần tiên quyết là gì?",
        "Các điều kiện để sinh viên được xét tốt nghiệp là gì?",
        "Hạng tốt nghiệp xuất sắc được tính dựa trên tiêu chí nào?"
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