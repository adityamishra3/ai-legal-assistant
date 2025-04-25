from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import json
import re
import ast
# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize chat model
chat = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama3-70b-8192"
)
add_on_prompt = """Note: The query will come in natural language, but since we need to pass it on to keywords based filtering, so modify the query and only keep the important key words and remove the helping words
    
"""
def get_intent(user_query: str) -> object:
    prompt = f"""
    You are a classifier and metadata extractor for a legal assistant chatbot used by lawyers.

    Given a user's legal query, do the following:

    1. Identify the **INTENT** of the query. Choose one of these:
    - "JUDGEMENT"
    - "LAW"
    - "SIMILAR_CASES"
    - "RAG_SEARCH"
    - "OTHER_SERVICES"

    2. Extract any **META INFORMATION** from the query, such as:
    - Query : {user_query}
    - Court (e.g., supremecourt, highcourts, maharastra, gujrat etc.)
    - Date Range (e.g., "mostrecent", "2021", "leastrecent" ,etc.)
    - Article/Section (e.g., IPC 376, Article 21)
    - Extra Keywords (e.g., bail, dowry, murder)

    Return the result in **valid JSON** format. Do not add any explanations or formatting.
    If any metadata is **not mentioned**, set its value to `null`.

    Now process this user query:

    \"\"\"{user_query}\"\"\"
    Example Input: 
        Show me all recent cases from allahbhad high court on IPC 376 and sexual assault
    Example Output:
    {{
        "intent": "JUDGEMENT",
        "meta_info": {{
            "query": "Show me all recent cases from allahbhad high court on IPC 376 and sexual assault",
            "court": "allahbad",
            "date_range": "mostrecent",
            "article_or_section": "IPC 376",
            "extra_keywords": ["sexual assault"]
        }}
    }}
    
    Example Input:
        what if i hurt religious feelings?
    Example Output:
    {{
        "intent": "LAW",
        "meta_info": {{
            "query": "what if i hurt religious feelings?",
            "court": null,
            "date_range": null,
            "article_or_section": null,
            "extra_keywords": ["religious feelings"]
        }}
    }}
    
    Return ONLY valid JSON with proper opening and closing braces. Do not include markdown, triple quotes, or any explanation. Your entire output must be a single JSON object. Use null instead of None for null values.
    """

    messages = [
        SystemMessage(content="You are a very intelligent AI Legal assistant who can identify Intents and key points in the query."),
        HumanMessage(content=prompt)
    ]

    response = chat.invoke(messages)
    response_content = response.content
    
    # Clean up common LLM formatting issues
    response_content = response_content.strip()
    
    # Remove any code block markers
    response_content = re.sub(r'^```json\s*', '', response_content)
    response_content = re.sub(r'\s*```$', '', response_content)
    
    # Replace Python None with JSON null if needed
    response_content = re.sub(r':\s*None', ': null', response_content)
    response_content = re.sub(r"'", '"', response_content)  # Replace single quotes with double quotes
    
    try:
        # Try to parse as JSON
        result = json.loads(response_content)
        return json.dumps(result)  # Return as a JSON string for consistency
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response: {response_content}")
        return json.dumps({
            "intent": "OTHER_SERVICES",
            "meta_info": {
                "query": user_query,
                "error": str(e),
                "court": None,
                "date_range": None,
                "article_or_section": None,
                "extra_keywords": []
            }
        }).replace("null", "None")  # Keep Python's None for consistency
def main(query=None):    
    query = "Give me information about state vs salman khan pls answer this"
    result = get_intent(query)
    print(result)
    return result

if __name__ == "__main__":
    main()
