from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat.rag_chatbot import RAGChatbot
from environment.rag_environment import RAGEnvironment
from algorithm.q_learning import QLearningAgent
from algorithm.dqn import DQNAgent
from algorithm.ddqn import DDQNAgent

app = FastAPI(
    title="RAG Chatbot API",
    description="API for RAG Chatbot with Reinforcement Learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatMessage(BaseModel):
    message: str
    algorithm: Optional[str] = "q_learning"  # "q_learning", "dqn", or "ddqn"

class ChatResponse(BaseModel):
    response: str
    action: str
    algorithm_used: str

class HealthResponse(BaseModel):
    status: str
    message: str

# Global variables to store trained agents
q_agent = None
dqn_agent = None
ddqn_agent = None
environment = None
chatbot_q = None
chatbot_dqn = None
chatbot_ddqn = None

def initialize_agents():
    """Initialize and train the agents"""
    global q_agent, dqn_agent, ddqn_agent, environment, chatbot_q, chatbot_dqn, chatbot_ddqn
    
    print("Initializing environment...")
    environment = RAGEnvironment()
    
    print("Training Q-Learning agent...")
    q_agent = QLearningAgent(environment.n_states, environment.n_actions)
    
    # Quick training for demo purposes (reduce episodes for faster startup)
    for episode in range(100):
        state = environment.reset()
        for step in range(20):
            action = q_agent.choose_action(state)
            next_state, reward, done = environment.step(action)
            q_agent.learn(state, action, reward, next_state, environment.gamma)
            state = next_state
            if done:
                break
    
    print("Training DQN agent...")
    # [Bước 1] Khởi tạo bộ nhớ lặp lại D — dùng để lưu trữ các trải nghiệm (experience replay)
    # [Bước 2] Khởi tạo hàm giá trị hành động Q với trọng số ngẫu nhiên θ — đây là mạng chính (main network)
    # [Bước 3] Khởi tạo mạng mục tiêu Q̂ (target network) với trọng số θ⁻ = θ — sao chép ban đầu từ mạng chính
    dqn_agent = DQNAgent(5, environment.n_actions, gamma=environment.gamma)
    
    # Quick training for demo purposes
    for episode in range(100):
    # [Bước 4] Khởi tạo trạng thái ban đầu s₁ của môi trường
    state = environment.reset()
    for step in range(20):
        # [Bước 5] Chọn hành động a_t theo chính sách ε-greedy dựa trên Q(φ_t, a; θ)
        action = dqn_agent.act(state)

        # [Bước 6] Thực hiện hành động a_t và quan sát phần thưởng r_t, trạng thái kế tiếp s_{t+1}
        next_state, reward, done = environment.step(action)

        # [Bước 7] Lưu chuyển tiếp (φ_t, a_t, r_t, φ_{t+1}) vào bộ nhớ D
        dqn_agent.remember(state, action, reward, next_state, done)

        # [Bước 8–11] Huấn luyện lại mạng bằng cơ chế Replay:
        #   - [Bước 8] Lấy mẫu minibatch ngẫu nhiên từ bộ nhớ D
        #   - [Bước 9] Tính giá trị mục tiêu y_j = r_j + γ * max_a' Q̂(φ_{j+1}, a'; θ⁻)
        #   - [Bước 10] Thực hiện lan truyền ngược (gradient descent)
        #                trên lỗi (y_j − Q(φ_j, a_j; θ))² để cập nhật θ
        #   - [Bước 11] Định kỳ cập nhật trọng số mạng mục tiêu θ⁻ ← θ
        dqn_agent.replay()

        # Cập nhật trạng thái
        state = next_state
        # Dừng sớm nếu tập đã kết thúc
        if done:
            break
    
    print("Training DDQN agent...")
    ddqn_agent = DDQNAgent(5, environment.n_actions)
    
    # Quick training for demo purposes
    for episode in range(100):
        state = environment.reset()
        for step in range(20):
            action = ddqn_agent.act(state)
            next_state, reward, done = environment.step(action)
            ddqn_agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        if episode % 10 == 0:
            ddqn_agent.replay()
    
    print("Creating chatbots...")
    chatbot_q = RAGChatbot(q_agent, environment)
    chatbot_dqn = RAGChatbot(dqn_agent, environment)
    chatbot_ddqn = RAGChatbot(ddqn_agent, environment)
    
    print("Agents initialized successfully!")

@app.on_event("startup")
async def startup_event():
    """Initialize agents when the server starts"""
    initialize_agents()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="RAG Chatbot API is running"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Main chat endpoint"""
    global chatbot_q, chatbot_dqn, chatbot_ddqn
    
    if not message.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        if message.algorithm.lower() == "dqn":
            if chatbot_dqn is None:
                raise HTTPException(status_code=503, detail="DQN agent not initialized")
            response, action = chatbot_dqn.respond(message.message)
            algorithm_used = "DQN"
        elif message.algorithm.lower() == "ddqn":
            if chatbot_ddqn is None:
                raise HTTPException(status_code=503, detail="DDQN agent not initialized")
            response, action = chatbot_ddqn.respond(message.message)
            algorithm_used = "DDQN"
        else:
            if chatbot_q is None:
                raise HTTPException(status_code=503, detail="Q-Learning agent not initialized")
            response, action = chatbot_q.respond(message.message)
            algorithm_used = "Q-Learning"
        
        return ChatResponse(
            response=response,
            action=action,
            algorithm_used=algorithm_used
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/algorithms")
async def get_available_algorithms():
    """Get list of available algorithms"""
    return {
        "algorithms": [
            {
                "name": "q_learning",
                "description": "Q-Learning Reinforcement Learning Algorithm",
                "available": chatbot_q is not None
            },
            {
                "name": "dqn",
                "description": "Deep Q-Network Algorithm", 
                "available": chatbot_dqn is not None
            },
            {
                "name": "ddqn",
                "description": "Double Deep Q-Network Algorithm",
                "available": chatbot_ddqn is not None
            }
        ]
    }

@app.post("/reinitialize")
async def reinitialize_agents():
    """Reinitialize and retrain agents"""
    try:
        initialize_agents()
        return {"message": "Agents reinitialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reinitializing agents: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
