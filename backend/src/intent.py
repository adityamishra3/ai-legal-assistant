from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize chat model
chat = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama3-70b-8192"
)

def get_intent(user_query: str)-> object:
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
    - Court (e.g., Supreme Court, High Court, state of maharastra, state of gujrat etc.)
    - Date Range (e.g., "mostrecent", "2021", "leastrecent" ,etc.)
    - Article/Section (e.g., IPC 376, Article 21)
    - Extra Keywords (e.g., bail, dowry, murder)

    Return the result in **valid JSON** format. Do not add any explanations or formatting.

    If any metadata is **not mentioned**, set its value to `None`.

    Now process this user query:

    \"\"\"{user_query}\"\"\"

    Example Output:
    ```json
    {{
        "intent": "JUDGEMENT",
        "meta_info": {{
            "query" : "Show me all recent Supreme Court cases on IPC 376 and sexual assault"
            "court": "Supreme Court",
            "date_range": "mostrecent",
            "article_or_section": "IPC 376",
            "extra_keywords": ["sexual assault"]
        }}
    }}
    ```
    """

    messages = [
        SystemMessage(content="You are a very intelligent AI Legal assistant who can identify Intents and key points in the query."),
        HumanMessage(content=prompt)
    ]

    response = chat.invoke(messages)
    return response.content

def main(query=None):    
    query = "Show me all the similar judgements K. M. Nanavati v. State of Maharashtra from 2024"
    result = get_intent(query)
    print(result)
    return result

if __name__ == "__main__":
    main()
