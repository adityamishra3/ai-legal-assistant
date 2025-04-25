import { useEffect, useState } from 'react';
import { FaPaperPlane } from 'react-icons/fa';

function QueryInput({ onSendQuery, isLoading, input }) {
  const [query, setQuery] = useState('');
  useEffect(()=>{
    setQuery(input)
  },[input])
  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSendQuery(query);
      setQuery('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="mt-auto">
      <div className="relative flex items-center">
        <textarea
          className="w-full px-4 py-3 pr-12 border border-neutral rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent resize-none"
          placeholder="Ask a legal question..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
          rows={2}
          disabled={isLoading}
        />
        <button
          type="submit"
          className={`absolute right-3 p-2 rounded-full ${
            query.trim() && !isLoading
              ? 'bg-primary text-white'
              : 'bg-neutral text-neutral-dark cursor-not-allowed'
          }`}
          disabled={!query.trim() || isLoading}
        >
          <FaPaperPlane />
        </button>
      </div>
      <p className="text-xs text-neutral-dark mt-1">
        Press Enter to send. Use Shift+Enter for a new line.
      </p>
    </form>
  );
}

export default QueryInput;