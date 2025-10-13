# RAG Chatbot with Reinforcement Learning

A Retrieval-Augmented Generation (RAG) chatbot system that uses Reinforcement Learning algorithms to optimize response strategies.

## Features

- **Multiple RL Algorithms**: Choose between Q-Learning, DQN (Deep Q-Network), and DDQN (Double Deep Q-Network)
- **RAG System**: Retrieval-Augmented Generation for accurate responses
- **FastAPI Backend**: High-performance REST API
- **React Frontend**: Modern, responsive user interface
- **Real-time Chat**: Interactive chat interface with algorithm selection

## Algorithms Supported

1. **Q-Learning**: Traditional tabular reinforcement learning
2. **DQN**: Deep Q-Network using neural networks
3. **DDQN**: Double Deep Q-Network with reduced overestimation bias (Latest addition!)

For detailed explanation of DDQN, see [DDQN_EXPLANATION.md](backend/algorithm/DDQN_EXPLANATION.md)

## How to Run

### Initialize Backend

```bash
cd backend
uv sync # In case not install dependencies yet
uv run api/main.py
```

The API will be available at `http://localhost:8000`

### Initialize Frontend

```bash
cd frontend
npm install # In case not install dependencies yet
npm start
```

The frontend will be available at `http://localhost:3000`

*Run in 2 different terminals please*

## API Documentation

Once the backend is running, access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

For more details, see [API_README.md](backend/API_README.md)

## Project Structure

```
RAG-DQN-QL/
├── backend/
│   ├── algorithm/           # RL algorithms (Q-Learning, DQN, DDQN)
│   ├── api/                # FastAPI endpoints
│   ├── chat/               # RAG chatbot logic
│   ├── environment/        # RL environment
│   ├── data/               # Training data
│   └── test/               # Test scripts
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   └── App.js         # Main app
│   └── public/
└── README.md
```

## Usage

1. Start both backend and frontend servers
2. Open http://localhost:3000 in your browser
3. Select an algorithm (Q-Learning, DQN, or DDQN) from the dropdown
4. Start chatting with the RAG chatbot
5. Compare performance across different algorithms

## Algorithm Comparison

Run the comparison script to see performance differences:

```bash
cd backend
python main.py
```

This will train all three algorithms and generate a detailed comparison chart.
