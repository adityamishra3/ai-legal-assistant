import { useState } from 'react';
import { FaAngleDown, FaAngleUp } from 'react-icons/fa';

function IntentDisplay({ metadata }) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!metadata || !metadata.response) return null;

  // Handle string responses
  if (typeof metadata.response === 'string') {
    return (
      <div className="mt-2 text-sm">
        <p className="font-medium">Response:</p>
        <p>{metadata.response}</p>
      </div>
    );
  }

  // Handle JSON responses
  const { intent, meta_info } = metadata.response;
  
  return (
    <div className="mt-2 text-sm border-t border-neutral pt-2">
      <div 
        className="flex items-center cursor-pointer text-primary-dark font-medium"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span>Details</span>
        {isExpanded ? <FaAngleUp className="ml-1" /> : <FaAngleDown className="ml-1" />}
      </div>
      
      {isExpanded && (
        <div className="bg-neutral-light p-3 rounded-md mt-2 space-y-2">
          <div>
            <span className="font-medium">Intent:</span> {intent}
          </div>
          
          {meta_info && (
            <div>
              <span className="font-medium">Meta Info:</span>
              <pre className="whitespace-pre-wrap bg-white p-2 rounded-md mt-1 text-xs overflow-x-auto">
                {JSON.stringify(meta_info, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default IntentDisplay;