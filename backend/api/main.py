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
from algorithm.sarsa import SarsaAgent

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
sar_agent = None
environment = None
chatbot_q = None
chatbot_dqn = None
chatbot_ddqn = None
chatbot_sar = None

def initialize_agents():
    """Initialize and train the agents"""
    global q_agent, dqn_agent, ddqn_agent, environment, chatbot_q, chatbot_dqn, chatbot_ddqn,sar_agent,chatbot_sar
    
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
    
    print("Training SARSA agent...")
    sar_agent = SarsaAgent(environment.n_states, environment.n_actions)
    # Quick training for demo purposes
    for episode in range(100):
        state = environment.reset()
        action = sar_agent.choose_action(state)
        total_reward = 0
        
        for step in range(20):
            next_state, reward, done = environment.step(action)
            next_action = sar_agent.choose_action(next_state)
            
            sar_agent.learn(state, action, reward, next_state, next_action, environment.gamma)
            
            state = next_state
            action = next_action
            total_reward += reward
            
            if done:
                break
        
     

            
    print("Training DQN agent...")
    dqn_agent = DQNAgent(5, environment.n_actions)
    
    # Quick training for demo purposes
    for episode in range(100):
        state = environment.reset()
        for step in range(20):
            action = dqn_agent.act(state)
            next_state, reward, done = environment.step(action)
            dqn_agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        if episode % 10 == 0:
            dqn_agent.replay()
    
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
    chatbot_sar = RAGChatbot(sar_agent, environment)
    
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
    global chatbot_q, chatbot_dqn, chatbot_ddqn, chatbot_sar
    
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
        elif message.algorithm.lower() == "sarsa":
            if chatbot_sar is None:
                raise HTTPException(status_code=503, detail="Sarsa agent not initialized")
            response, action = chatbot_sar.respond(message.message)
            algorithm_used = "SARSA"
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
            },
            {
                "name": "sarsa",
                "description": "SARSA Reinforcement Learning Algorithm", 
                "available": chatbot_sar is not None
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


