import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

interface Message {
  id: number;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: string;
}

interface ChatResponse {
  response: string;
  user_id: string;
  session_id: string;
  timestamp: string;
  status: string;
  metadata: any;
}

const UserIdInput: React.FC<{ onSubmit: (userId: string) => void }> = ({ onSubmit }) => {
  const [inputUserId, setInputUserId] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputUserId.trim()) {
      onSubmit(inputUserId.trim());
    }
  };

  return (
    <div className="user-id-modal">
      <div className="user-id-card">
        <h2>🔐 Welcome to Umbranet Governor</h2>
        <p>Enter your User ID to continue your conversation or start a new one.</p>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={inputUserId}
            onChange={(e) => setInputUserId(e.target.value)}
            placeholder="Enter your User ID (e.g., alex, john_doe, etc.)"
            className="user-id-input"
            autoFocus
          />
          <button 
            type="submit" 
            disabled={!inputUserId.trim()}
            className="user-id-submit"
          >
            Continue
          </button>
        </form>
        <div className="user-id-info">
          <p><strong>💡 Tip:</strong> Use the same User ID to continue previous conversations.</p>
          <p><strong>🧠 Memory:</strong> Your AI remembers everything across sessions!</p>
        </div>
      </div>
    </div>
  );
};

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [userId, setUserId] = useState('');
  const [sessionId] = useState(`session_${Date.now()}`);
  const [isUserIdSet, setIsUserIdSet] = useState(false);
  const [showUserIdInput, setShowUserIdInput] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize user ID from localStorage or show input
  useEffect(() => {
    const savedUserId = localStorage.getItem('umbranet_user_id');
    if (savedUserId) {
      setUserId(savedUserId);
      setIsUserIdSet(true);
    } else {
      setShowUserIdInput(true);
    }
  }, []);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ 
        behavior: 'smooth',
        block: 'end',
        inline: 'nearest'
      });
    }
  };

  useEffect(() => {
    // Small delay to ensure DOM is updated before scrolling
    const timer = setTimeout(() => {
      scrollToBottom();
    }, 100);
    
    return () => clearTimeout(timer);
  }, [messages]);

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading || !isUserIdSet || !userId) return;

    const userMessage: Message = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await axios.post<ChatResponse>('/api/chat', {
        message: inputValue,
        user_id: userId,
        session_id: sessionId,
      });

      const assistantMessage: Message = {
        id: Date.now() + 1,
        text: response.data.response,
        sender: 'assistant',
        timestamp: response.data.timestamp,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage: Message = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error processing your message. Please try again.',
        sender: 'assistant',
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleUserIdSubmit = (inputUserId: string) => {
    const trimmedUserId = inputUserId.trim();
    if (trimmedUserId) {
      setUserId(trimmedUserId);
      setIsUserIdSet(true);
      setShowUserIdInput(false);
      localStorage.setItem('umbranet_user_id', trimmedUserId);
    }
  };

  const handleChangeUser = () => {
    setShowUserIdInput(true);
    setIsUserIdSet(false);
    setMessages([]); // Clear messages when switching users
  };

  const handleLogout = () => {
    localStorage.removeItem('umbranet_user_id');
    setUserId('');
    setIsUserIdSet(false);
    setShowUserIdInput(true);
    setMessages([]);
  };

  return (
    <div className="app">
      {showUserIdInput && (
        <UserIdInput onSubmit={handleUserIdSubmit} />
      )}
      
      <header className="app-header">
        <div className="header-content">
          <div className="header-title">
            <h1>🧠 Umbranet Governor</h1>
            <p>Headless AI Operating System</p>
          </div>
          {isUserIdSet && (
            <div className="user-controls">
              <div className="current-user">
                <span className="user-label">👤 {userId}</span>
              </div>
              <div className="user-actions">
                <button onClick={handleChangeUser} className="change-user-btn" title="Switch User">
                  🔄
                </button>
                <button onClick={handleLogout} className="logout-btn" title="Logout">
                  🚪
                </button>
              </div>
            </div>
          )}
        </div>
      </header>

      {isUserIdSet && (
        <div className="chat-container">
          <div className="messages-container">
            {messages.length === 0 && (
              <div className="welcome-message">
                <h3>Welcome back, {userId}!</h3>
                <p>Your personal AI assistant with persistent memory across conversations.</p>
                <p>I remember our previous conversations. Start typing to continue...</p>
              </div>
            )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={`message ${message.sender === 'user' ? 'user-message' : 'assistant-message'}`}
            >
              <div className="message-content">
                {message.text}
              </div>
              <div className="message-timestamp">
                {new Date(message.timestamp).toLocaleTimeString()}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message assistant-message loading">
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

        <div className="input-container">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message here... (Press Enter to send)"
            className="message-input"
            rows={3}
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={!inputValue.trim() || isLoading}
            className="send-button"
          >
            {isLoading ? '⏳' : '📤'}
          </button>
          </div>
        </div>
      )}

      {isUserIdSet && (
        <footer className="app-footer">
          <p>Session: {sessionId}</p>
          <p>🧠 4-Tier Memory: Redis • PostgreSQL • Neo4j • Procedural</p>
        </footer>
      )}
    </div>
  );
};

export default App;