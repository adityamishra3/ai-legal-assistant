import { useState } from 'react';
import { FaChartBar, FaSearch, FaGavel, FaBook } from 'react-icons/fa';

function StatsDisplay() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="fixed bottom-4 right-4">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="bg-secondary text-white p-3 rounded-full shadow-lg"
      >
        <FaChartBar />
      </button>
      
      {isOpen && (
        <div className="absolute bottom-14 right-0 bg-white rounded-lg shadow-xl border border-neutral p-4 w-72">
          <h3 className="font-bold text-lg mb-3">System Statistics</h3>
          
          <div className="space-y-3">
            <div className="flex items-center">
              <div className="p-2 bg-blue-100 rounded-full mr-3">
                <FaSearch className="text-blue-600" />
              </div>
              <div>
                <div className="text-sm text-neutral-dark">RAG Searches</div>
                <div className="font-medium">24 queries</div>
              </div>
            </div>
            
            <div className="flex items-center">
              <div className="p-2 bg-purple-100 rounded-full mr-3">
                <FaGavel className="text-purple-600" />
              </div>
              <div>
                <div className="text-sm text-neutral-dark">Judgments Referenced</div>
                <div className="font-medium">12 cases</div>
              </div>
            </div>
            
            <div className="flex items-center">
              <div className="p-2 bg-green-100 rounded-full mr-3">
                <FaBook className="text-green-600" />
              </div>
              <div>
                <div className="text-sm text-neutral-dark">Law Sections Explained</div>
                <div className="font-medium">8 sections</div>
              </div>
            </div>
          </div>
          
          <div className="mt-4 pt-3 border-t border-neutral text-xs text-neutral-dark">
            Demo data for presentation purposes only
          </div>
        </div>
      )}
    </div>
  );
}

export default StatsDisplay;