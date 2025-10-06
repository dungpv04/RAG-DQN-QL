import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './ChatInterface.css';

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      content: 'Xin chào! Tôi là trợ lý chatbot RAG của bạn. Tôi có thể giúp gì cho bạn hôm nay?',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('q_learning');
  const [algorithms, setAlgorithms] = useState([]);
  const [serverStatus, setServerStatus] = useState('unknown');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Check server health and load algorithms on component mount
    checkHealth();
    loadAlgorithms();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await axios.get('/health');
      setServerStatus('healthy');
    } catch (error) {
      console.error('Health check failed:', error);
      setServerStatus('unhealthy');
    }
  };

  const loadAlgorithms = async () => {
    try {
      const response = await axios.get('/algorithms');
      setAlgorithms(response.data.algorithms);
    } catch (error) {
      console.error('Failed to load algorithms:', error);
    }
  };

  const reinitializeAgents = async () => {
    setIsLoading(true);
    try {
      await axios.post('/reinitialize');
      
      const successMessage = {
        id: Date.now(),
        type: 'bot',
        content: 'Các agent đã được khởi tạo lại thành công! Bạn có thể tiếp tục trò chuyện.',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, successMessage]);
      await loadAlgorithms(); // Reload algorithms after reinitialization
    } catch (error) {
      console.error('Failed to reinitialize agents:', error);
      const errorMessage = {
        id: Date.now(),
        type: 'bot',
        content: 'Không thể khởi tạo lại các agent. Vui lòng thử lại.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post('/chat', {
        message: userMessage.content,
        algorithm: selectedAlgorithm
      });

      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: response.data.response,
        timestamp: new Date(),
        algorithm: response.data.algorithm_used,
        action: response.data.action
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: 'Xin lỗi, tôi gặp lỗi. Vui lòng thử lại.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([
      {
        id: 1,
        type: 'bot',
        content: 'Xin chào! Tôi là trợ lý chatbot RAG của bạn. Tôi có thể giúp gì cho bạn hôm nay?',
        timestamp: new Date()
      }
    ]);
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <div className="header-content">
          <div className="title-section">
            <h1>RAG Chatbot</h1>
            <span className="subtitle">AI Assistant with Retrieval-Augmented Generation</span>
            <div className="status-indicator">
              <span className={`status-dot ${serverStatus}`}></span>
              Máy chủ: {serverStatus === 'healthy' ? 'hoạt động' : serverStatus === 'unhealthy' ? 'lỗi' : 'không xác định'}
            </div>
          </div>
          <div className="header-controls">
            <div className="algorithm-selector">
              <label htmlFor="algorithm">Thuật toán:</label>
              <select 
                id="algorithm"
                value={selectedAlgorithm} 
                onChange={(e) => setSelectedAlgorithm(e.target.value)}
                disabled={isLoading}
              >
                {algorithms.map(algo => (
                  <option key={algo.name} value={algo.name} disabled={!algo.available}>
                    {algo.name.toUpperCase()} {!algo.available ? '(không khả dụng)' : ''}
                  </option>
                ))}
              </select>
            </div>
            <button className="action-button" onClick={reinitializeAgents} disabled={isLoading}>
              🔄 Khởi tạo lại
            </button>
            <button className="clear-button" onClick={clearChat}>
              🧹 Xóa chat
            </button>
          </div>
        </div>
      </div>

      <div className="chat-messages">
        {messages.map((message) => (
          <div key={message.id} className={`message ${message.type}`}>
            <div className="message-avatar">
              {message.type === 'user' ? (
                <div className="user-avatar">U</div>
              ) : (
                <div className="bot-avatar">🤖</div>
              )}
            </div>
            <div className="message-content">
              <div className="message-text">{message.content}</div>
              <div className="message-meta">
                {message.algorithm && (
                  <span className="algorithm-badge">{message.algorithm}</span>
                )}
                {message.action && (
                  <span className="action-badge">Action: {message.action}</span>
                )}
                <div className="message-time">
                  {message.timestamp.toLocaleTimeString([], { 
                    hour: '2-digit', 
                    minute: '2-digit' 
                  })}
                </div>
              </div>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="message bot">
            <div className="message-avatar">
              <div className="bot-avatar">🤖</div>
            </div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input">
        <div className="input-container">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Nhập tin nhắn của bạn tại đây..."
            disabled={isLoading}
            rows="1"
          />
          <button 
            onClick={sendMessage} 
            disabled={!inputMessage.trim() || isLoading}
            className="send-button"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path 
                d="M7 11L12 6L17 11M12 18V7" 
                stroke="currentColor" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round"
                transform="rotate(90 12 12)"
              />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
