# RAG Chatbot Frontend

A modern React-based chat interface for the RAG (Retrieval-Augmented Generation) chatbot.

## Features

- 🎨 Modern ChatGPT-like interface
- 💬 Real-time messaging
- 🤖 Beautiful bot responses with typing indicators
- 📱 Responsive design for mobile and desktop
- 🧹 Clear chat functionality
- ⚡ Fast and smooth animations

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

### Backend Integration

Make sure your FastAPI backend is running on `http://localhost:8000` before starting the frontend. The React app is configured to proxy API requests to the backend.

## Available Scripts

- `npm start` - Runs the app in development mode
- `npm build` - Builds the app for production
- `npm test` - Launches the test runner
- `npm eject` - Ejects from Create React App (one-way operation)

## Project Structure

```
src/
├── components/
│   ├── ChatInterface.js      # Main chat component
│   └── ChatInterface.css     # Chat styling
├── App.js                    # Root component
├── App.css                   # App styling
├── index.js                  # Entry point
└── index.css                 # Global styles
```

## API Integration

The frontend communicates with the FastAPI backend through:

- `POST /api/chat` - Send message and receive bot response

## Customization

### Styling
- Modify `ChatInterface.css` for chat interface styling
- Update `index.css` for global styles
- Colors and gradients can be customized in the CSS files

### Features
- Add message history persistence
- Implement file upload functionality
- Add voice input/output
- Customize bot avatar and responses

## Browser Support

This project supports all modern browsers including:
- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)
