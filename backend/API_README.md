# RAG Chatbot FastAPI Backend

This is a FastAPI backend for the RAG Chatbot with Reinforcement Learning system. It provides REST API endpoints to interact with the chatbot using either Q-Learning or DQN algorithms.

## Features

- **Chat Endpoint**: Send messages and receive responses from the RAG chatbot
- **Algorithm Selection**: Choose between Q-Learning and DQN algorithms
- **Health Check**: Monitor API status
- **CORS Support**: Ready for frontend integration

## Installation

1. Install dependencies:
```bash
pip install -e .
```

Or with uv:
```bash
uv pip install -e .
```

## Running the Server

### Development Mode
```bash
# From the project root directory
python app/main.py
```

Or using uvicorn directly:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

## API Endpoints

### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "RAG Chatbot API is running"
}
```

### 2. Chat
```http
POST /chat
```

**Request Body:**
```json
{
  "message": "Quy chế đào tạo đại học quy định những gì?",
  "algorithm": "q_learning"  // Optional: "q_learning" or "dqn"
}
```

**Response:**
```json
{
  "response": "Chatbot response here...",
  "action": "search_documents",
  "algorithm_used": "Q-Learning"
}
```

### 3. Available Algorithms
```http
GET /algorithms
```

**Response:**
```json
{
  "algorithms": [
    {
      "name": "q_learning",
      "description": "Q-Learning Reinforcement Learning Algorithm",
      "available": true
    },
    {
      "name": "dqn", 
      "description": "Deep Q-Network Algorithm",
      "available": true
    }
  ]
}
```

### 4. Reinitialize Agents
```http
POST /reinitialize
```

**Response:**
```json
{
  "message": "Agents reinitialized successfully"
}
```

## Testing the API

Use the provided test client:
```bash
python test_api_client.py
```

Or test manually with curl:
```bash
# Health check
curl http://localhost:8000/health

# Send a chat message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quy chế đào tạo đại học quy định những gì?", "algorithm": "q_learning"}'
```

## Project Structure

```
app/
├── main.py              # FastAPI application
├── __init__.py
├── api/                 # API routes (for future expansion)
├── core/                # Core configuration
├── db/                  # Database models
├── models/              # Pydantic models
├── schemas/             # API schemas
└── services/            # Business logic services

test_api_client.py       # Test client for the API
```

## Error Handling

The API includes comprehensive error handling:
- **400**: Bad Request (empty message)
- **500**: Internal Server Error (processing errors)
- **503**: Service Unavailable (agents not initialized)

## CORS Configuration

CORS is configured to allow all origins for development. For production, update the `allow_origins` parameter in `app/main.py`.

## Notes

- The agents are trained with reduced episodes (100) for faster startup
- For production use, consider:
  - Pre-training agents and saving models
  - Adding authentication
  - Implementing rate limiting
  - Adding logging and monitoring
  - Configuring CORS appropriately
