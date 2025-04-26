import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.docstore.document import Document
from document_processor import store_judgments, query_judgments_with_llm, get_llm_response
import os
import time
import uuid


def search_indian_kanoon(intent: dict, limit=20):
    query = intent["meta_info"].get("query")
    doctype = intent["meta_info"].get("court")
    date_range = intent["meta_info"].get("date_range")
    '''if type(date_range) is int:
        date_range = f"year%3A+{intent['meta_info'].get("date_range")}"
    else:
        date_range = f"sortby%3A{intent['meta_info'].get(date_range)}"'''

    search_query = query.replace(" ", "+")
    url = f"https://indiankanoon.org/search/?formInput={search_query}+doctypes:{doctype}+{date_range}"
    print("URL: ",url)
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")
    snip = []

    results = []
    for i, result_block in enumerate(soup.select(".result")):
        if i >= limit:
            break

        a_tag = result_block.find("a")
        title = a_tag.get_text(strip=True)
        full_url = "https://indiankanoon.org" + a_tag["href"]

        headline_div = result_block.find_next("div", class_="headline")
        snippet = headline_div.get_text(strip=True) if headline_div else None
        snip.append(snippet)

        hlbottom = result_block.find_next("div", class_="hlbottom")
        court = hlbottom.select_one(".docsource").get_text(strip=True) if hlbottom else None
        
        results.append({
            "title": title.strip(),
            "url": full_url.strip(),
            "snippet": snippet.strip() if snippet else None,
            "court": court.strip() if court else None
        })

    system_prompt = "You will determine the most relevant snippet based on the query and only retuen the index location of the result in the array."
    user_prompt = f"Select the most relevant result based on: {query}. Return the index location (starts from 0) of the most relevant reults from: {results}. Only return the number as I need to extract it from the response."

    index = get_llm_response(system_prompt, user_prompt)
    print("Index: ", index)

    return results, index


def fetch_full_judgment(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")
    doc_div = soup.find("div", class_="judgments")
    return doc_div.get_text(strip=True, separator="\n") if doc_div else ""


def summarize_large_judgment(judgment_text, query, chunk_size=3000):
    # Split judgment into manageable chunks
    chunks = [judgment_text[i:i+chunk_size] for i in range(0, len(judgment_text), chunk_size)]
    
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        system_prompt = """You are a legal assistant. Concisely extract key information from this judgment segment. Focus only on facts, legal principles, and reasoning visible in this segment."""
        
        user_prompt = f"""This is segment {i+1}/{len(chunks)} of a judgment related to: "{query}"
        
        Extract only the important legal information from this segment:
        
        {chunk}"""
        
        chunk_summary = get_llm_response(system_prompt, user_prompt)
        chunk_summaries.append(chunk_summary)
    
    # Final pass to combine chunk summaries
    system_prompt = """You are a legal assistant specializing in Indian law. Create a cohesive summary from these judgment segment summaries, focusing on key facts, legal issues, court reasoning, and relevance to the query."""
    
    user_prompt = f"""Based on the query "{query}", create a unified summary from these judgment segment summaries:
    
    {' '.join(chunk_summaries)}
    
    Provide a structured final summary that highlights the most relevant aspects related to the query. Return the output only in text format (no markdown or any other format) and it should only state the information, no need for generic comments."""
    
    final_summary = get_llm_response(system_prompt, user_prompt)
    return final_summary

def main():
    intent = {
        "intent": "SIMILAR_CASES",
        "meta_info": {
            "query": "Show me all the similar judgements like State vs Salman Khan",
            "court": "State of Maharashtra",
            "date_range": "mostrecent",
            "article_or_section": None,
            "extra_keywords": []
        }
    }
    session_id = uuid.uuid1()
    print(session_id)
    print("Searching Indian Kanoon...")
    search_results, index = search_indian_kanoon(intent)
    index = int(index)

    search_results = search_results[index]

    docs = []
    print("Scraping full judgments...")
    '''for case in search_results:
        print(f"➡️ {case['title']}")'''
    judgment_text = fetch_full_judgment(search_results['url']).replace("  "," ").replace("\n\n","\n")
    '''    time.sleep(2) 
        print("Judgement Text: ", judgment_text )
        docs.append({
            "title": case["title"],
            "url": case["url"],
            "court": case["court"],
            "snippet": case["snippet"],
            "judgment_text": judgment_text
        })'''
    
    #print("Judgement Result: ")
    #TODO PASS JUDGEMENT TO LLM, OPTIMIZE QUERY BEING SENT TO INDIAN KANOON, LLM OUTPUT ONLY IN TEXT

    # Use the summarize_large_judgment function instead of direct LLM call
    result = summarize_large_judgment(judgment_text, intent['meta_info']['query'])
    print("Final Result: ", result)
    return result

if __name__ == "__main__":
    main()
