"""
Test script để kiểm tra DDQN integration với API
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.ddqn import DDQNAgent
from environment.rag_environment import RAGEnvironment
from chat.rag_chatbot import RAGChatbot

def test_ddqn_agent_creation():
    """Test khởi tạo DDQN agent"""
    print("=" * 60)
    print("TEST 1: Khởi tạo DDQN Agent")
    print("=" * 60)
    
    try:
        agent = DDQNAgent(state_size=5, action_size=5)
        print("✅ DDQN Agent được khởi tạo thành công!")
        print(f"   - State size: {agent.state_size}")
        print(f"   - Action size: {agent.action_size}")
        print(f"   - Epsilon: {agent.epsilon}")
        print(f"   - Memory size: {len(agent.memory)}")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo DDQN Agent: {e}")
        return False

def test_ddqn_training():
    """Test training DDQN agent"""
    print("\n" + "=" * 60)
    print("TEST 2: Training DDQN Agent")
    print("=" * 60)
    
    try:
        env = RAGEnvironment()
        agent = DDQNAgent(5, env.n_actions)
        
        print(f"Training với {10} episodes...")
        
        for episode in range(10):
            state = env.reset()
            total_reward = 0
            
            for step in range(20):
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            if episode % 5 == 0:
                agent.replay()
                print(f"   Episode {episode}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.4f}")
        
        print("✅ DDQN Agent training thành công!")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi training DDQN Agent: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ddqn_chatbot():
    """Test DDQN với RAG chatbot"""
    print("\n" + "=" * 60)
    print("TEST 3: DDQN với RAG Chatbot")
    print("=" * 60)
    
    try:
        env = RAGEnvironment()
        agent = DDQNAgent(5, env.n_actions)
        
        # Quick training
        print("Quick training...")
        for episode in range(20):
            state = env.reset()
            for step in range(10):
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
            if episode % 5 == 0:
                agent.replay()
        
        # Test chatbot
        chatbot = RAGChatbot(agent, env)
        
        test_queries = [
            "Quy chế đào tạo đại học quy định những gì?",
            "Thời gian học tập tối đa là bao lâu?",
            "Điều kiện tốt nghiệp là gì?"
        ]
        
        print("\nTest chatbot responses:")
        for query in test_queries:
            response, action = chatbot.respond(query)
            print(f"\n📝 Query: {query}")
            print(f"   Action: {action}")
            print(f"   Response: {response[:100]}...")
        
        print("\n✅ DDQN Chatbot hoạt động thành công!")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi test DDQN Chatbot: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ddqn_statistics():
    """Test DDQN statistics"""
    print("\n" + "=" * 60)
    print("TEST 4: DDQN Statistics")
    print("=" * 60)
    
    try:
        agent = DDQNAgent(5, 5)
        env = RAGEnvironment()
        
        # Training
        for episode in range(50):
            state = env.reset()
            for step in range(10):
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
            agent.replay()
        
        # Get statistics
        stats = agent.get_statistics()
        
        print("Statistics:")
        print(f"   - Epsilon: {stats['epsilon']:.4f}")
        print(f"   - Memory size: {stats['memory_size']}")
        print(f"   - Average loss: {stats['avg_loss']:.4f}")
        print(f"   - Total training steps: {stats['total_training_steps']}")
        
        print("\n✅ DDQN Statistics hoạt động thành công!")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi test DDQN Statistics: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ddqn_model_save_load():
    """Test save và load DDQN model"""
    print("\n" + "=" * 60)
    print("TEST 5: DDQN Model Save/Load")
    print("=" * 60)
    
    try:
        import tempfile
        
        # Create and train agent
        agent1 = DDQNAgent(5, 5)
        env = RAGEnvironment()
        
        for episode in range(10):
            state = env.reset()
            for step in range(10):
                action = agent1.act(state)
                next_state, reward, done = env.step(action)
                agent1.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
            agent1.replay()
        
        epsilon_before = agent1.epsilon
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name
        
        agent1.save_model(tmp_path)
        print(f"✅ Model saved to {tmp_path}")
        
        # Load model
        agent2 = DDQNAgent(5, 5)
        agent2.load_model(tmp_path)
        print(f"✅ Model loaded from {tmp_path}")
        
        # Verify
        assert abs(agent2.epsilon - epsilon_before) < 0.0001, "Epsilon không khớp!"
        print(f"   - Epsilon before: {epsilon_before:.4f}")
        print(f"   - Epsilon after: {agent2.epsilon:.4f}")
        
        # Cleanup
        os.remove(tmp_path)
        
        print("\n✅ DDQN Model Save/Load hoạt động thành công!")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi test DDQN Model Save/Load: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "🔬" * 30)
    print("DDQN API INTEGRATION TEST SUITE")
    print("🔬" * 30 + "\n")
    
    tests = [
        test_ddqn_agent_creation,
        test_ddqn_training,
        test_ddqn_chatbot,
        test_ddqn_statistics,
        test_ddqn_model_save_load
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 TẤT CẢ TESTS ĐỀU PASS! DDQN ĐÃ ĐƯỢC TÍCH HỢP THÀNH CÔNG!")
    else:
        print("\n⚠️ MỘT SỐ TESTS FAILED. VUI LÒNG KIỂM TRA LẠI!")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

