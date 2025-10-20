import numpy as np
import matplotlib.pyplot as plt
from chat.rag_chatbot import RAGChatbot
from environment.rag_environment import RAGEnvironment
from algorithm.q_learning import QLearningAgent
from algorithm.dqn import DQNAgent
from algorithm.ddqn import DDQNAgent
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
        
        if episode % 1000 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Q-Learning Episode {episode}, Avg Reward: {avg_reward:.2f}")
    
    return agent, rewards_history

def train_dqn(episodes=1000):
    """Huấn luyện DQN"""
    env = RAGEnvironment()
    # Truyền gamma từ môi trường và để agent tự cập nhật target theo số bước
    agent = DQNAgent(5, env.n_actions, gamma=env.gamma)  # 5 features as input
    
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(50):
            # [Step 5] following ε-greedy policy, select a_t — chọn hành động ε-greedy
            action = agent.act(state)
            # [Step 6] execute a_t in emulator and observe r_t, s_{t+1} — tương tác môi trường
            next_state, reward, done = env.step(action)
            # [Step 7] store transition (φ_t, a_t, r_t, φ_{t+1}) in D — lưu vào replay
            agent.remember(state, action, reward, next_state, done)
            # [Steps 8–10] sample minibatch and gradient step — huấn luyện trong replay()
            agent.replay()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        rewards_history.append(total_reward)
        
        if episode % 1000 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"DQN Episode {episode}, Avg Reward: {avg_reward:.2f}")
    
    return agent, rewards_history

def train_ddqn(episodes=1000):
    """Huấn luyện Double Deep Q-Network (DDQN)"""
    env = RAGEnvironment()
    agent = DDQNAgent(5, env.n_actions)  # 5 features as input
    
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
        
        if episode % 1000 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"DDQN Episode {episode}, Avg Reward: {avg_reward:.2f}")
    
    return agent, rewards_history

# ========================= Demo và Đánh giá =========================
def compare_algorithms():
    """So sánh hiệu quả của Q-Learning, DQN và DDQN"""
    print("=== Bắt đầu huấn luyện Q-Learning ===")
    q_agent, q_rewards = train_q_learning(episodes=10000)
    
    print("\n=== Bắt đầu huấn luyện DQN ===")
    dqn_agent, dqn_rewards = train_dqn(episodes=10000)
    
    print("\n=== Bắt đầu huấn luyện DDQN ===")
    ddqn_agent, ddqn_rewards = train_ddqn(episodes=10000)
    
    # Vẽ biểu đồ so sánh chi tiết
    plt.figure(figsize=(16, 12))
    
    # 1. Raw rewards comparison
    plt.subplot(2, 3, 1)
    plt.plot(q_rewards, label='Q-Learning', alpha=0.6, color='blue')
    plt.plot(dqn_rewards, label='DQN', alpha=0.6, color='red')
    plt.plot(ddqn_rewards, label='DDQN', alpha=0.6, color='green')
    plt.title('Reward theo Episode (Raw)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Moving average comparison
    window = 50
    q_ma = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
    dqn_ma = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
    ddqn_ma = np.convolve(ddqn_rewards, np.ones(window)/window, mode='valid')
    
    plt.subplot(2, 3, 2)
    plt.plot(q_ma, label='Q-Learning (MA)', linewidth=2, color='blue')
    plt.plot(dqn_ma, label='DQN (MA)', linewidth=2, color='red')
    plt.plot(ddqn_ma, label='DDQN (MA)', linewidth=2, color='green')
    plt.title(f'Moving Average Reward (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Performance comparison (last 100 episodes)
    plt.subplot(2, 3, 3)
    final_q = np.mean(q_rewards[-100:])
    final_dqn = np.mean(dqn_rewards[-100:])
    final_ddqn = np.mean(ddqn_rewards[-100:])
    
    algorithms = ['Q-Learning', 'DQN', 'DDQN']
    performance = [final_q, final_dqn, final_ddqn]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    
    bars = plt.bar(algorithms, performance, color=colors)
    plt.title('Performance Cuối Cùng (100 episodes cuối)')
    plt.ylabel('Average Reward')
    
    # Add value labels on bars
    for bar, value in zip(bars, performance):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    # 4. Learning curve comparison (cumulative reward)
    plt.subplot(2, 3, 4)
    q_cumulative = np.cumsum(q_rewards)
    dqn_cumulative = np.cumsum(dqn_rewards)
    ddqn_cumulative = np.cumsum(ddqn_rewards)
    
    plt.plot(q_cumulative, label='Q-Learning', color='blue')
    plt.plot(dqn_cumulative, label='DQN', color='red')
    plt.plot(ddqn_cumulative, label='DDQN', color='green')
    plt.title('Cumulative Reward')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Convergence analysis (rolling standard deviation)
    plt.subplot(2, 3, 5)
    def rolling_std(data, window=50):
        return [np.std(data[max(0, i-window):i+1]) for i in range(len(data))]
    
    q_std = rolling_std(q_rewards)
    dqn_std = rolling_std(dqn_rewards)
    ddqn_std = rolling_std(ddqn_rewards)
    
    plt.plot(q_std, label='Q-Learning', color='blue')
    plt.plot(dqn_std, label='DQN', color='red')
    plt.plot(ddqn_std, label='DDQN', color='green')
    plt.title('Stability Analysis (Rolling Std)')
    plt.xlabel('Episode')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    plt.subplot(2, 3, 6)
    plt.text(0.05, 0.95, 'PERFORMANCE SUMMARY', fontsize=14, fontweight='bold', 
             transform=plt.gca().transAxes)
    
    plt.text(0.05, 0.85, f'Q-Learning:', fontsize=12, fontweight='bold', 
             color='blue', transform=plt.gca().transAxes)
    plt.text(0.05, 0.80, f'  Final Avg: {final_q:.3f}', fontsize=10, 
             transform=plt.gca().transAxes)
    plt.text(0.05, 0.75, f'  Total: {q_cumulative[-1]:.1f}', fontsize=10,
             transform=plt.gca().transAxes)
    
    plt.text(0.05, 0.65, f'DQN:', fontsize=12, fontweight='bold', 
             color='red', transform=plt.gca().transAxes)
    plt.text(0.05, 0.60, f'  Final Avg: {final_dqn:.3f}', fontsize=10,
             transform=plt.gca().transAxes)
    plt.text(0.05, 0.55, f'  Total: {dqn_cumulative[-1]:.1f}', fontsize=10,
             transform=plt.gca().transAxes)
    plt.text(0.05, 0.50, f'  vs Q-Learning: {((final_dqn-final_q)/final_q*100):.1f}%', 
             fontsize=10, transform=plt.gca().transAxes)
    
    plt.text(0.05, 0.40, f'DDQN:', fontsize=12, fontweight='bold', 
             color='green', transform=plt.gca().transAxes)
    plt.text(0.05, 0.35, f'  Final Avg: {final_ddqn:.3f}', fontsize=10,
             transform=plt.gca().transAxes)
    plt.text(0.05, 0.30, f'  Total: {ddqn_cumulative[-1]:.1f}', fontsize=10,
             transform=plt.gca().transAxes)
    plt.text(0.05, 0.25, f'  vs Q-Learning: {((final_ddqn-final_q)/final_q*100):.1f}%', 
             fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.20, f'  vs DQN: {((final_ddqn-final_dqn)/final_dqn*100):.1f}%', 
             fontsize=10, transform=plt.gca().transAxes)
    
    # Best performing algorithm
    best_alg = algorithms[np.argmax(performance)]
    plt.text(0.05, 0.10, f'🏆 Best: {best_alg}', fontsize=12, fontweight='bold',
             color='orange', transform=plt.gca().transAxes)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.suptitle('So sánh Thuật toán: Q-Learning vs DQN vs DDQN', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('algorithm_comparison_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed comparison
    print("\n" + "="*60)
    print("📊 KẾT QUẢ SO SÁNH CHI TIẾT")
    print("="*60)
    print(f"Q-Learning - Final Average: {final_q:.3f}")
    print(f"DQN        - Final Average: {final_dqn:.3f} ({((final_dqn-final_q)/final_q*100):+.1f}%)")
    print(f"DDQN       - Final Average: {final_ddqn:.3f} ({((final_ddqn-final_q)/final_q*100):+.1f}%)")
    print(f"\n🎯 DDQN vs DQN Improvement: {((final_ddqn-final_dqn)/final_dqn*100):+.1f}%")
    print(f"🏆 Best Algorithm: {best_alg}")
    print("="*60)
    
    return q_agent, dqn_agent, ddqn_agent

# ========================= Giả định các hàm phụ trợ =========================

# ========================= Ứng dụng Chatbot =========================


# ========================= Main Execution =========================
if __name__ == "__main__":
    print("=== RAG Chatbot với Reinforcement Learning ===\n")
    
    # So sánh thuật toán
    q_agent, dqn_agent, ddqn_agent = compare_algorithms()
    
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
    
    print("=== Demo Chatbot với DDQN ===")
    chatbot_ddqn = RAGChatbot(ddqn_agent, env)
    
    for query in sample_queries:
        response, action = chatbot_ddqn.respond(query)
        print(f"Câu hỏi: {query}")
        print(f"Hành động: {action}")
        print(f"Trả lời: {response}\n")
    
    # Thống kê cuối cùng
    print("="*60)
    print("🎯 SUMMARY: So sánh 3 thuật toán đã hoàn thành!")
    print("📊 Biểu đồ chi tiết đã được lưu: algorithm_comparison_detailed.png")
    print("🤖 Demo chatbot với cả 3 agents đã hoàn thành!")
    print("="*60)