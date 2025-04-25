from flask import Flask, request, jsonify
from src.intent import get_intent 
from src.document_processor import query_constitution_with_llm, query_bns_with_llm
import json
from flask_cors import CORS

# from judgement_handler import handle_judgment_search
# from rag_handler import handle_rag_search
app = Flask(__name__)
CORS(app)

@app.route("/legal-query", methods=["POST"])
def main_handler():
    query = request.json.get("query")

    if not query:
        return jsonify({"error": "Missing 'query' in request body"}), 400
    
    try:
        # Get intent data as a string
        intent_data_str = get_intent(query)
        
        # Replace None with null for proper JSON parsing
        intent_data_str = intent_data_str.replace("None", "null")
        
        # Parse JSON
        intent_data = json.loads(intent_data_str)
        
        print("🔍 Intent Identified:", intent_data)
        intent = intent_data.get("intent")
        meta_info = intent_data.get("meta_info")
        
        # Handle different intents
        if intent in ["JUDGEMENT", "SIMILAR_CASES"]:
            result = {
                "status": "processing",
                "message": f"Searching for relevant judgments related to: {query}",
                "intent_type": intent,
                "query_details": meta_info
            }
        elif intent in ["LAW", "RAG_SEARCH"]:
            # Enhanced placeholder response
            result = query_bns_with_llm(meta_info["query"])
        else:
            result = {
                "status": "unimplemented",
                "message": "This service is not yet implemented or categorized under 'OTHER_SERVICES'.",
                "intent_type": intent,
                "query_details": meta_info
            }
        
        return jsonify({"response": result}), 200
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "response": f"Error processing: {query}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
