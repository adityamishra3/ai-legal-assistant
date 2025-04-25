import { useState, useEffect } from 'react';
import Header from './components/Header';
import ChatInterface from './components/ChatInterface';
import QueryInput from './components/QueryInput';
import StatsDisplay from './components/StatsDisplay';
import axios from 'axios';

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [input, setInput] = useState("")

  const handleSendQuery = async (query) => {
    if (!query.trim()) return;
    
    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      text: query,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    
    try {
      const response = await axios.post('http://localhost:5000/legal-query', {
        query: query
      });
      console.log("Response:", response)
      // Add bot response to chat
      const botMessage = {
        id: Date.now() + 1,
        text: typeof response.data.response === 'string' 
          ? response.data.response 
          : JSON.stringify(response.data.response, null, 2),
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString()
      };
      console.log(botMessage)
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending query:', error);
      
      // Add error message
      const errorMessage = {
        id: Date.now() + 1,
        text: 'Sorry, there was an error processing your request. Please try again.',
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
        isError: true
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen">
      <Header />
      <main className="flex-1 container mx-auto px-4 py-6 overflow-hidden flex flex-col">
        <ChatInterface messages={messages} isLoading={isLoading} setInput={setInput}/>
        <QueryInput onSendQuery={handleSendQuery} isLoading={isLoading} input={input} />
      </main>
      {/* <StatsDisplay /> */}
    </div>
  );
}

export default App;