import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.docstore.document import Document
import os
import time
import uuid


def search_indian_kanoon(intent: dict, limit=10):
    query = intent["meta_info"].get("query")
    doctype = intent["meta_info"].get("court")
    date_range = intent["meta_info"].get("date_range")
    if type(date_range) is int:
        date_range = f"year%3A+{date_range}"
    else:
        date_range = f"sortby%3A{date_range}"
    search_query = query.replace(" ", "+")
    url = f"https://indiankanoon.org/search/?formInput={search_query}+doctypes:{doctype}+{date_range}"
    print("URL: ",url)
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")

    results = []
    for i, result_block in enumerate(soup.select(".result")):
        if i >= limit:
            break

        a_tag = result_block.find("a")
        title = a_tag.get_text(strip=True)
        full_url = "https://indiankanoon.org" + a_tag["href"]

        headline_div = result_block.find_next("div", class_="headline")
        snippet = headline_div.get_text(strip=True) if headline_div else None

        hlbottom = result_block.find_next("div", class_="hlbottom")
        court = hlbottom.select_one(".docsource").get_text(strip=True) if hlbottom else None
        
        results.append({
            "title": title.strip(),
            "url": full_url.strip(),
            "snippet": snippet.strip() if snippet else None,
            "court": court.strip() if court else None
        })

    return results


def fetch_full_judgment(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")
    doc_div = soup.find("div", class_="judgments")
    return doc_div.get_text(strip=True, separator="\n") if doc_div else ""



def main():
    intent = {
        "intent": "SIMILAR_CASES",
        "meta_info": {
            "query": "Show me all the similar judgements K. M. Nanavati v. State of Maharashtra from 2024",
            "court": "State of Maharashtra",
            "date_range": "mostrecent",
            "article_or_section": None,
            "extra_keywords": []
        }
    }
    session_id = uuid.uuid1()
    print(session_id)
    print("Searching Indian Kanoon...")
    search_results = search_indian_kanoon(intent)

    docs = []
    print("Scraping full judgments...")
    for case in search_results:
        print(f"➡️ {case['title']}")
        judgment_text = fetch_full_judgment(case['url']).replace("  "," ").replace("\n\n","\n")
        time.sleep(2) 
        print("Judgement Text: ", judgment_text )
        docs.append({
            "title": case["title"],
            "url": case["url"],
            "court": case["court"],
            "snippet": case["snippet"],
            "judgment_text": judgment_text
        })
        # break
    # print("Embedding & storing in FAISS...")
    # embed_and_store(docs)


if __name__ == "__main__":
    main()
