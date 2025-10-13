# Demo và so sánh DDQN với DQN trong RAG system
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add algorithm directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algorithm'))

try:
    from ddqn import DDQNAgent, create_rag_ddqn_agent, compare_dqn_vs_ddqn
    from dqn import DQNAgent  # Import DQN để so sánh
except ImportError:
    print("Warning: PyTorch libraries not installed. This is a demo file.")
    print("To run this demo, install: pip install torch numpy matplotlib")


class RAGEnvironment:
    """
    Simulated RAG Environment để test DDQN vs DQN
    """
    def __init__(self):
        self.state_space_size = 243  # 3^5 = 243 possible states
        self.action_space_size = 5   # 5 possible actions
        self.current_state = 0
        self.episode_length = 50
        self.step_count = 0
        
        # Action names cho visualization
        self.action_names = [
            "Quote directly",     # 0: Trích dẫn trực tiếp
            "Summarize",         # 1: Tóm tắt
            "Explain",           # 2: Diễn giải
            "Clarify",           # 3: Hỏi lại để làm rõ
            "Fallback"           # 4: Thừa nhận không biết
        ]
    
    def reset(self):
        """Reset environment về state ban đầu"""
        self.current_state = np.random.randint(0, self.state_space_size)
        self.step_count = 0
        return self.current_state
    
    def step(self, action):
        """
        Thực hiện action và trả về (next_state, reward, done)
        """
        self.step_count += 1
        
        # Decode current state để tính reward
        features = self._decode_state(self.current_state)
        reward = self._calculate_reward(features, action)
        
        # Generate next state
        next_state = np.random.randint(0, self.state_space_size)
        
        # Check if episode done
        done = self.step_count >= self.episode_length
        
        self.current_state = next_state
        return next_state, reward, done
    
    def _decode_state(self, state):
        """Decode state thành 5 features"""
        features = []
        temp_state = state
        for i in range(5):
            features.append(temp_state % 3)
            temp_state //= 3
        return features
    
    def _calculate_reward(self, features, action):
        """
        Tính reward dựa trên features và action
        Reward cao khi action phù hợp với context
        """
        similarity, confidence, context, ambiguity, scope = features
        
        # Reward matrix dựa trên domain knowledge của RAG
        if action == 0:  # Quote directly
            if confidence >= 2 and similarity >= 2:
                return 1.0
            else:
                return -0.5
        
        elif action == 1:  # Summarize
            if context >= 1 and scope >= 1:
                return 0.8
            else:
                return -0.2
        
        elif action == 2:  # Explain
            if confidence >= 1 and ambiguity <= 1:
                return 0.9
            else:
                return -0.3
        
        elif action == 3:  # Clarify
            if ambiguity >= 2 or confidence <= 0:
                return 0.7
            else:
                return -0.4
        
        else:  # Fallback
            if confidence <= 0 and similarity <= 0:
                return 0.5
            else:
                return -0.6


def train_agent(agent, env, episodes=1000, update_target_freq=100):
    """
    Training function cho agent
    
    Args:
        agent: DQN hoặc DDQN agent
        env: RAG environment
        episodes: Số episodes training
        update_target_freq: Tần suất update target network
    
    Returns:
        list: Episode rewards history
    """
    rewards_history = []
    losses_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            # Chọn action
            action = agent.act(state)
            
            # Thực hiện action
            next_state, reward, done = env.step(action)
            
            # Lưu experience
            agent.remember(state, action, reward, next_state, done)
            
            # Training
            agent.replay()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Update target network
        if episode % update_target_freq == 0:
            agent.update_target_network()
        
        rewards_history.append(total_reward)
        
        # Logging
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return rewards_history


def compare_algorithms():
    """
    So sánh performance của DQN vs DDQN
    """
    print("=== Training và So sánh DQN vs DDQN ===")
    
    # Tạo environment
    env = RAGEnvironment()
    
    # Tạo agents
    try:
        dqn_agent = DQNAgent(state_size=5, action_size=5, lr=0.001)
        ddqn_agent = DDQNAgent(state_size=5, action_size=5, lr=0.001)
        
        print("Training DQN...")
        dqn_rewards = train_agent(dqn_agent, env, episodes=500)
        
        print("Training DDQN...")
        ddqn_rewards = train_agent(ddqn_agent, env, episodes=500)
        
        # Visualization
        plt.figure(figsize=(12, 8))
        
        # Plot rewards
        plt.subplot(2, 2, 1)
        plt.plot(dqn_rewards, label='DQN', alpha=0.7)
        plt.plot(ddqn_rewards, label='DDQN', alpha=0.7)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        
        # Plot moving average
        plt.subplot(2, 2, 2)
        window = 50
        dqn_ma = [np.mean(dqn_rewards[max(0, i-window):i+1]) for i in range(len(dqn_rewards))]
        ddqn_ma = [np.mean(ddqn_rewards[max(0, i-window):i+1]) for i in range(len(ddqn_rewards))]
        
        plt.plot(dqn_ma, label='DQN (MA)', linewidth=2)
        plt.plot(ddqn_ma, label='DDQN (MA)', linewidth=2)
        plt.title(f'Moving Average Rewards (window={window})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        
        # Statistics comparison
        plt.subplot(2, 2, 3)
        final_dqn = np.mean(dqn_rewards[-100:])
        final_ddqn = np.mean(ddqn_rewards[-100:])
        
        algorithms = ['DQN', 'DDQN']
        performance = [final_dqn, final_ddqn]
        colors = ['lightblue', 'lightgreen']
        
        bars = plt.bar(algorithms, performance, color=colors)
        plt.title('Final Performance (Last 100 episodes)')
        plt.ylabel('Average Reward')
        
        # Add value labels on bars
        for bar, value in zip(bars, performance):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Summary statistics
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.8, f"DQN Final Avg: {final_dqn:.3f}", fontsize=12)
        plt.text(0.1, 0.7, f"DDQN Final Avg: {final_ddqn:.3f}", fontsize=12)
        plt.text(0.1, 0.6, f"Improvement: {((final_ddqn-final_dqn)/final_dqn*100):.1f}%", fontsize=12)
        plt.text(0.1, 0.4, f"DQN Final ε: {dqn_agent.epsilon:.3f}", fontsize=10)
        plt.text(0.1, 0.3, f"DDQN Final ε: {ddqn_agent.epsilon:.3f}", fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Training Summary')
        
        plt.tight_layout()
        plt.savefig('dqn_vs_ddqn_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nKết quả cuối cùng:")
        print(f"DQN Average Reward: {final_dqn:.3f}")
        print(f"DDQN Average Reward: {final_ddqn:.3f}")
        print(f"DDQN improvement: {((final_ddqn-final_dqn)/final_dqn*100):.1f}%")
        
    except Exception as e:
        print(f"Cannot run training comparison: {e}")
        print("This requires PyTorch installation.")


def demonstrate_ddqn_features():
    """
    Demo các tính năng đặc biệt của DDQN
    """
    print("=== DDQN Features Demonstration ===")
    
    try:
        # Tạo DDQN agent
        agent = create_rag_ddqn_agent()
        
        print("1. DDQN Agent Configuration:")
        print(f"   - State size: {agent.state_size}")
        print(f"   - Action size: {agent.action_size}")
        print(f"   - Memory buffer size: {agent.memory.maxlen}")
        print(f"   - Initial epsilon: {agent.epsilon}")
        print()
        
        print("2. State Encoding/Decoding Demo:")
        test_states = [0, 121, 242]  # Low, medium, high encoded states
        for state in test_states:
            tensor = agent.state_to_tensor(state)
            print(f"   State {state} -> Features: {tensor.squeeze().tolist()}")
        print()
        
        print("3. Action Selection Demo:")
        for state in test_states:
            action = agent.act(state)
            print(f"   State {state} -> Action: {action}")
        print()
        
        print("4. Experience Replay Demo:")
        # Add some dummy experiences
        for i in range(10):
            agent.remember(i, i%5, np.random.random(), (i+1)%243, i==9)
        
        print(f"   Memory size after adding experiences: {len(agent.memory)}")
        
        # Try replay
        agent.replay(batch_size=5)
        print("   Replay training completed successfully")
        print()
        
        print("5. Target Network Update Demo:")
        print("   Before update:")
        old_target_param = list(agent.target_network.parameters())[0][0][0].item()
        print(f"   Sample target network parameter: {old_target_param:.6f}")
        
        agent.update_target_network()
        
        print("   After update:")
        new_target_param = list(agent.target_network.parameters())[0][0][0].item()
        print(f"   Sample target network parameter: {new_target_param:.6f}")
        print()
        
        print("6. Statistics:")
        stats = agent.get_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"Cannot run DDQN demonstration: {e}")
        print("This requires PyTorch installation.")


def main():
    """
    Main function để chạy all demos
    """
    print("🤖 DDQN Implementation for RAG System")
    print("=" * 50)
    
    # Basic comparison
    compare_dqn_vs_ddqn()
    print()
    
    # DDQN features demo
    demonstrate_ddqn_features()
    print()
    
    # Training comparison (nếu có PyTorch)
    compare_algorithms()
    
    print("\n✅ Demo hoàn thành!")
    print("File ddqn.py đã được tạo với full implementation của DDQN.")
    print("Bạn có thể sử dụng nó cho bài thuyết trình về so sánh DQN vs DDQN.")


if __name__ == "__main__":
    main()