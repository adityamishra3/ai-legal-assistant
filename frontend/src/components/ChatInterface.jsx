import { useEffect, useRef } from 'react';
import { FaUser, FaRobot } from 'react-icons/fa';
import IntentDisplay from './IntentDisplay';

function ChatInterface({ messages, isLoading, setInput}) {
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto pb-4 mb-4">
      {messages.length === 0 ? (
        <div className="h-full flex flex-col items-center justify-center text-neutral-dark">
          <FaRobot className="text-6xl mb-4 text-primary" />
          <h2 className="text-2xl font-semibold mb-2">Welcome to Legal Assistant</h2>
          <p className="text-center max-w-md">
            Ask questions about laws, judgments, or legal advice. The AI will analyze your query and provide relevant information.
          </p>
          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl">
            {[
              "Show me all recent cases from allahbhad high court on IPC 376 and sexual assault",
              "Give me information about state vs salman khan",
              "Tell me about acts against which there is no right of private defence?",
              "A group attacks a person due to religion. Which provisions of BNS handle this?"
            ].map((suggestion, index) => (
              <button 
                key={index}
                className="bg-neutral-light hover:bg-neutral border border-neutral rounded-lg px-4 py-2 text-left"
                onClick={() => {
                  setInput(suggestion)
                }}
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {messages.map((message) => (
            <div 
              key={message.id} 
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div 
                className={`max-w-3xl rounded-lg px-4 py-3 ${
                  message.sender === 'user' 
                    ? 'bg-primary text-white rounded-br-none' 
                    : message.isError 
                      ? 'bg-red-50 text-red-600 border border-red-200 rounded-bl-none' 
                      : 'bg-white shadow-sm border border-neutral rounded-bl-none'
                }`}
              >
                <div className="flex items-center mb-1">
                  {message.sender === 'user' ? (
                    <FaUser className="mr-2" />
                  ) : (
                    <FaRobot className="mr-2" />
                  )}
                  <span className="text-xs opacity-70">{message.timestamp}</span>
                </div>
                <div>
                  {message.sender === 'bot' && typeof message.text === 'string' && message.text.startsWith('{') ? (
                    <pre className="whitespace-pre-wrap text-sm">
                      {message.text}
                    </pre>
                  ) : (
                    <p className="whitespace-pre-wrap">{message.text}</p>
                  )}
                  
                  {message.sender === 'bot' && message.metadata && (
                    <IntentDisplay metadata={message.metadata} />
                  )}
                </div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white shadow-sm border border-neutral rounded-lg rounded-bl-none px-4 py-3 max-w-3xl">
                <div className="flex items-center">
                  <FaRobot className="mr-2" />
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-neutral-dark rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 bg-neutral-dark rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    <div className="w-2 h-2 bg-neutral-dark rounded-full animate-bounce" style={{ animationDelay: '600ms' }}></div>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      )}
    </div>
  );
}

export default ChatInterface;