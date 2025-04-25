import { FaBalanceScale } from 'react-icons/fa';

function Header() {
  return (
    <header className="bg-primary text-white py-4 shadow-md">
      <div className="container mx-auto px-4 flex items-center">
        <FaBalanceScale className="text-2xl mr-3" />
        <h1 className="text-xl font-bold">Legal Assistant</h1>
        {/* <span className="text-xs bg-white text-primary px-2 py-1 rounded-full ml-3">RAG-powered</span> */}
        <div className="text-xs bg-white text-primary px-2 py-1 rounded-full ml-auto">
        RAG-powered
        </div>
      </div>
    </header>
  );
}

export default Header;