from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import re
import os
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize chat model
chat = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama3-70b-8192"
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY")
)

# Preprocessing functions
def clean_text(text):
    """Clean and normalize text for better chunking and retrieval"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep legal references
    text = re.sub(r'[^\w\s\.\,\;\:\(\)\[\]\{\}\-\'\"\§]', '', text)
    return text.strip()

def extract_legal_metadata(text):
    """Extract potential legal metadata from text"""
    metadata = {}
    
    # Try to identify sections, articles, chapters
    section_match = re.search(r'Section\s+(\d+[A-Za-z]*)', text)
    if section_match:
        metadata["section"] = section_match.group(1)
    
    article_match = re.search(r'Article\s+(\d+[A-Za-z]*)', text)
    if article_match:
        metadata["article"] = article_match.group(1)
    
    chapter_match = re.search(r'Chapter\s+(\d+[A-Za-z]*)', text)
    if chapter_match:
        metadata["chapter"] = chapter_match.group(1)
        
    return metadata

def bns():
    db = Chroma(persist_directory="./bns_db", embedding_function=embeddings)

    pdf_path = "backend/src/data/bns.pdf"

    if not os.path.exists(pdf_path):
        print("❌ File not found:", pdf_path)
    else:
        print("✅ File found:", pdf_path)

    loader = PyMuPDFLoader(pdf_path)
    doc = loader.load()

    # Improved text splitter with better parameters for legal documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Increased chunk size for better context
        chunk_overlap=100,  # Increased overlap to maintain context between chunks
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],  # Better splitting hierarchy
        is_separator_regex=False
    )

    # Load and preprocess chunks
    raw_chunks = loader.load()
    processed_chunks = []
    
    for chunk in raw_chunks:
        # Clean text
        cleaned_text = clean_text(chunk.page_content)
        chunk.page_content = cleaned_text
        
        # Extract and add legal metadata
        legal_metadata = extract_legal_metadata(cleaned_text)
        chunk.metadata.update(legal_metadata)
        
        processed_chunks.append(chunk)
    
    # Split the processed chunks
    chunks = text_splitter.split_documents(processed_chunks)
    chunk_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunk_with_ids if chunk.metadata.get("id") not in existing_ids]

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        print("Documents added successfully")
    else:
        print("No new documents to add.")
        
    stored_data = db.get(include=["embeddings"])
    print(f"Stored Data in DB: {stored_data}")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0
    
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown_source")
        page = chunk.metadata.get("page", "unknown_page")
        
        # Include extracted legal metadata in ID if available
        section = chunk.metadata.get("section", "")
        article = chunk.metadata.get("article", "")
        chapter = chunk.metadata.get("chapter", "")
        
        metadata_suffix = ""
        if section:
            metadata_suffix += f"_s{section}"
        if article:
            metadata_suffix += f"_a{article}"
        if chapter:
            metadata_suffix += f"_c{chapter}"
        
        current_page_id = f"{source}{page}"
        
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
            
        chunk_id = f"{current_page_id}:{current_chunk_index}{metadata_suffix}"
        # Add the ID to the chunk's metadata
        chunk.metadata["id"] = chunk_id
        
        last_page_id = current_page_id
    
    return chunks

def query_bns(user_query, top_k=10):
    """
    Search for relevant document chunks based on user query
    
    Args:
        user_query (str): The user's question or query
        top_k (int): Number of most relevant chunks to return (default: 10)
    
    Returns:
        list: List of most relevant document chunks with their content and metadata
    """
    # Initialize the Chroma database with the same embeddings
    db = Chroma(persist_directory="../bns_db", embedding_function=embeddings)
    
    # Clean the query for better matching
    clean_query = clean_text(user_query)
    
    # Generate query variations for better retrieval
    query_variations = [
        clean_query,
        f"Find information about {clean_query}",
        f"What does the BNS say about {clean_query}?",
        f"Legal provisions related to {clean_query}"
    ]
    
    # First pass: Get a broader set of results
    all_results = []
    for query_var in query_variations:
        results = db.similarity_search_with_score(
            query=query_var,
            k=top_k // 2  # Get half the total from each query variation
        )
        all_results.extend(results)
    
    # Remove duplicates
    unique_results = []
    seen_ids = set()
    for doc, score in all_results:
        doc_id = doc.metadata.get("id", "")
        if doc_id not in seen_ids:
            unique_results.append((doc, score))
            seen_ids.add(doc_id)
    
    # Sort by relevance score
    sorted_results = sorted(unique_results, key=lambda x: x[1])
    
    # Take top_k unique results
    top_results = sorted_results[:top_k]
    
    # Format results for easier consumption
    formatted_results = []
    for doc, score in top_results:
        # Simple reranking: boost scores for documents with query terms in them
        boost = 0
        query_terms = clean_query.lower().split()
        for term in query_terms:
            if term in doc.page_content.lower():
                boost += 0.05  # Small boost for each query term
        
        adjusted_score = score - boost  # Lower score is better in this case
        
        formatted_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": adjusted_score
        })
    
    # Sort again by adjusted score
    formatted_results = sorted(formatted_results, key=lambda x: x["relevance_score"])
    
    return formatted_results

def store_judgments(judgment_documents):
    """
    Process and store judgment documents in a Chroma vector database
    
    Args:
        judgment_documents (list): List of dictionaries containing judgment information
                                  Each dict should have keys: title, url, court, snippet, judgment_text
    
    Returns:
        int: Number of new documents added to the database
    """
    # Initialize Chroma DB for judgments
    judgment_db = Chroma(persist_directory="./judgments_db", embedding_function=embeddings)
    
    # Convert the documents to LangChain Document format
    langchain_docs = []
    for doc in judgment_documents:
        # Clean the judgment text
        cleaned_text = clean_text(doc["judgment_text"])
        
        langchain_docs.append(
            Document(
                page_content=cleaned_text,
                metadata={
                    "title": doc["title"],
                    "url": doc["url"],
                    "court": doc["court"],
                    "snippet": doc["snippet"]
                }
            )
        )
    
    # Improved text splitter for judgments
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", ";", ",", " ", ""],
        is_separator_regex=False
    )
    
    chunks = []
    for doc in langchain_docs:
        doc_chunks = text_splitter.split_documents([doc])
        chunks.extend(doc_chunks)
    
    # Calculate chunk IDs
    processed_chunks = calculate_judgment_chunk_ids(chunks)
    
    # Check for existing documents to avoid duplicates
    existing_items = judgment_db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing judgments in DB: {len(existing_ids)}")
    
    # Filter out chunks that already exist in the database
    new_chunks = [chunk for chunk in processed_chunks if chunk.metadata.get("id") not in existing_ids]
    
    # Add new chunks to the database
    if len(new_chunks):
        print(f"Adding new judgment chunks: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        judgment_db.add_documents(new_chunks, ids=new_chunk_ids)
        print("Judgment chunks added successfully")
    else:
        print("No new judgment chunks to add.")
    
    return len(new_chunks)

def calculate_judgment_chunk_ids(chunks):
    """
    Calculate unique IDs for judgment document chunks
    
    Args:
        chunks (list): List of Document objects
    
    Returns:
        list: Updated Document objects with IDs in metadata
    """
    # Group chunks by title to track position within each judgment
    judgment_chunks = {}
    
    for chunk in chunks:
        title = chunk.metadata.get("title", "unknown_title")
        court = chunk.metadata.get("court", "unknown_court")
        
        if title not in judgment_chunks:
            judgment_chunks[title] = []
        
        judgment_chunks[title].append(chunk)
    
    # Generate sequential IDs for chunks within each judgment
    for title, title_chunks in judgment_chunks.items():
        clean_title = "".join(c if c.isalnum() else "_" for c in title[:30])
        
        for i, chunk in enumerate(title_chunks):
            # Extract any case citation or paragraph numbers in the chunk
            citation_match = re.search(r'\((\d{4})\)\s+(\d+)\s+SCC', chunk.page_content)
            para_match = re.search(r'paragraph\s+(\d+)', chunk.page_content.lower())
            
            citation_suffix = ""
            if citation_match:
                year, report = citation_match.groups()
                citation_suffix = f"_{year}_{report}"
            
            para_suffix = ""
            if para_match:
                para = para_match.group(1)
                para_suffix = f"_p{para}"
            
            chunk_id = f"{clean_title}:{i}{citation_suffix}{para_suffix}"
            chunk.metadata["id"] = chunk_id
            chunk.metadata["position"] = i  # Add position metadata
            
            # Add extracted metadata for better retrieval
            if citation_match:
                chunk.metadata["year"] = citation_match.group(1)
                chunk.metadata["report_number"] = citation_match.group(2)
            
            if para_match:
                chunk.metadata["paragraph"] = para_match.group(1)
    
    # Flatten the dictionary back to a list
    result_chunks = []
    for chunks_list in judgment_chunks.values():
        result_chunks.extend(chunks_list)
    
    return result_chunks

def query_judgments(user_query, top_k=10):
    """
    Search for relevant judgment chunks based on user query
    
    Args:
        user_query (str): The user's question or query
        top_k (int): Number of most relevant chunks to return (default: 10)
    
    Returns:
        list: List of most relevant judgment chunks with their content and metadata
    """
    # Initialize the Chroma database with the same embeddings
    judgment_db = Chroma(persist_directory="./judgments_db", embedding_function=embeddings)
    
    # Clean and process the query
    clean_query = clean_text(user_query)
    
    # Generate multiple search queries for better coverage
    query_variations = [
        clean_query,
        f"legal judgment about {clean_query}",
        f"court ruling on {clean_query}",
        f"legal precedent for {clean_query}"
    ]
    
    # Collect results from all query variations
    all_results = []
    for query_var in query_variations:
        # Use similarity search since MMR might not be available in all versions
        results = judgment_db.similarity_search_with_score(
            query=query_var,
            k=top_k // 2
        )
        all_results.extend(results)
    
    # Remove duplicates
    unique_results = []
    seen_ids = set()
    for doc, score in all_results:
        doc_id = doc.metadata.get("id", "")
        if doc_id not in seen_ids:
            unique_results.append((doc, score))
            seen_ids.add(doc_id)
    
    # Sort by relevance score
    sorted_results = sorted(unique_results, key=lambda x: x[1])
    
    # Take top_k unique results
    top_results = sorted_results[:top_k]
    
    # Format results for easier consumption
    formatted_results = []
    for doc, score in top_results:
        # Extract year if available for citation
        year = doc.metadata.get("year", "")
        year_boost = 0
        if year:
            # Add a small boost for more recent judgments
            try:
                year_int = int(year)
                current_year = 2025  # Current year
                recency = min(10, max(0, (current_year - year_int))) / 10
                year_boost = recency * 0.1  # Small boost based on recency
            except ValueError:
                pass
        
        adjusted_score = score - year_boost
        
        formatted_results.append({
            "content": doc.page_content,
            "title": doc.metadata.get("title", "Unknown"),
            "court": doc.metadata.get("court", "Unknown"),
            "url": doc.metadata.get("url", ""),
            "snippet": doc.metadata.get("snippet", ""),
            "year": year,
            "relevance_score": adjusted_score
        })
    
    # Final sorting by adjusted score
    formatted_results = sorted(formatted_results, key=lambda x: x["relevance_score"])
    
    return formatted_results

def old_constitution():
    """
    Process and store constitution document with rate limiting to avoid hitting API limits
    """
    db = Chroma(persist_directory="./constitution_db", embedding_function=embeddings)

    pdf_path = "backend/src/data/constitution.pdf"

    if not os.path.exists(pdf_path):
        print("❌ File not found:", pdf_path)
        return
    else:
        print("✅ File found:", pdf_path)

    loader = PyMuPDFLoader(pdf_path)
    raw_doc = loader.load()
    
    # Clean and preprocess the documents
    processed_docs = []
    for doc in raw_doc:
        # Clean text
        cleaned_text = clean_text(doc.page_content)
        doc.page_content = cleaned_text
        
        # Extract constitutional metadata
        article_matches = re.findall(r'Article\s+(\d+[A-Za-z]*)', cleaned_text)
        if article_matches:
            doc.metadata["articles"] = article_matches
        
        part_match = re.search(r'Part\s+(\w+)', cleaned_text)
        if part_match:
            doc.metadata["part"] = part_match.group(1)
        
        schedule_match = re.search(r'Schedule\s+(\w+)', cleaned_text)
        if schedule_match:
            doc.metadata["schedule"] = schedule_match.group(1)
        
        processed_docs.append(doc)

    # Improved text splitter for constitutional documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", ";", "—", " ", ""],
        is_separator_regex=False
    )

    chunks = text_splitter.split_documents(processed_docs)
    print(f"Total chunks created: {len(chunks)}")
    
    # Calculate chunk IDs
    chunk_with_ids = calculate_constitution_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in constitution DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunk_with_ids if chunk.metadata.get("id") not in existing_ids]
    
    if len(new_chunks) == 0:
        print("No new documents to add.")
        return
    
    print(f"Adding new documents: {len(new_chunks)}")
    
    # Process in smaller batches with time delays to avoid rate limits
    batch_size = 30  # Adjust based on your rate limits
    total_batches = (len(new_chunks) + batch_size - 1) // batch_size  # Ceiling division
    
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i+batch_size]
        batch_ids = [chunk.metadata["id"] for chunk in batch]
        
        current_batch = (i // batch_size) + 1
        print(f"Processing batch {current_batch}/{total_batches} ({len(batch)} chunks)")
        
        db.add_documents(batch, ids=batch_ids)
        
        # Sleep between batches to avoid hitting rate limits
        if i + batch_size < len(new_chunks):
            print(f"Waiting for rate limit reset...")
            time.sleep(10)  # Wait 10 seconds between batches, adjust as needed
    
    print("All constitution document chunks added successfully")

def calculate_constitution_chunk_ids(chunks):
    """
    Calculate unique IDs for constitution document chunks
    
    Args:
        chunks (list): List of Document objects
    
    Returns:
        list: Updated Document objects with IDs in metadata
    """
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown_source")
        page = chunk.metadata.get("page", "unknown_page")
        
        # Add metadata identifiers to ID
        articles = chunk.metadata.get("articles", [])
        part = chunk.metadata.get("part", "")
        schedule = chunk.metadata.get("schedule", "")
        
        metadata_suffix = ""
        if articles and len(articles) > 0:
            metadata_suffix += f"_a{articles[0]}"
        if part:
            metadata_suffix += f"_part{part}"
        if schedule:
            metadata_suffix += f"_sched{schedule}"
        
        chunk_id = f"constitution_{source}{page}:{i}{metadata_suffix}"
        chunk.metadata["id"] = chunk_id
        
    return chunks

def query_constitution(user_query, top_k=10):
    """
    Search for relevant constitution document chunks based on user query
    
    Args:
        user_query (str): The user's question or query
        top_k (int): Number of most relevant chunks to return (default: 10)
    
    Returns:
        list: List of most relevant constitution document chunks with their content and metadata
    """
    # Initialize the Chroma database with the same embeddings
    constitution_db = Chroma(persist_directory="../constitution_db", embedding_function=embeddings)
    
    # Clean and process the query
    clean_query = clean_text(user_query)
    
    # Extract potential article references
    article_match = re.search(r'article\s+(\d+[a-z]*)', clean_query.lower())
    part_match = re.search(r'part\s+(\w+)', clean_query.lower())
    schedule_match = re.search(r'schedule\s+(\w+)', clean_query.lower())
    
    # Query filtering for explicit references
    filter_dict = None
    if article_match or part_match or schedule_match:
        filter_dict = {}
        if article_match:
            article_num = article_match.group(1)
            filter_dict["articles"] = {"$contains": article_num}
        if part_match:
            part_name = part_match.group(1)
            filter_dict["part"] = part_name
        if schedule_match:
            schedule_name = schedule_match.group(1)
            filter_dict["schedule"] = schedule_name
    
    # Try with filter first, then without if needed
    if filter_dict:
        filtered_results = constitution_db.similarity_search_with_score(
            query=clean_query,
            k=top_k,
            filter=filter_dict
        )
        
        # If no results with filter, fall back to regular search
        if not filtered_results:
            filter_dict = None
    
    # Generate query variations for better coverage
    query_variations = [
        clean_query,
        f"constitutional provision about {clean_query}",
        f"constitution of india on {clean_query}",
        f"what does the constitution say about {clean_query}"
    ]
    
    # If no filter or filter returned no results, use multiple queries
    results = []
    if not filter_dict:
        all_results = []
        for query_var in query_variations:
            var_results = constitution_db.similarity_search_with_score(
                query=query_var,
                k=top_k // 2
            )
            all_results.extend(var_results)
            
        # Remove duplicates
        seen_ids = set()
        for doc, score in all_results:
            doc_id = doc.metadata.get("id", "")
            if doc_id not in seen_ids:
                results.append((doc, score))
                seen_ids.add(doc_id)
                
        # Sort and limit
        results = sorted(results, key=lambda x: x[1])[:top_k]
    else:
        results = filtered_results
    
    # Format results for easier consumption
    formatted_results = []
    for doc, score in results:
        # Apply a simple reranking based on keyword matching
        boost = 0
        query_terms = clean_query.lower().split()
        for term in query_terms:
            if term in doc.page_content.lower():
                boost += 0.05
                
        # Extra boost for article match
        if article_match and "articles" in doc.metadata:
            article_num = article_match.group(1)
            if article_num in doc.metadata["articles"]:
                boost += 0.1
                
        adjusted_score = score - boost
        
        formatted_results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": adjusted_score
        })
    
    # Sort by adjusted score
    formatted_results = sorted(formatted_results, key=lambda x: x["relevance_score"])
    
    return formatted_results

def get_llm_response(system_prompt, user_prompt):
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = chat.invoke(messages)
    return response.content

def query_bns_with_llm(user_query, top_k=20):
    """
    Search for relevant BNS document chunks and generate an LLM response
    
    Args:
        user_query (str): The user's question about the Bharatiya Nyaya Sanhita
        top_k (int): Number of most relevant chunks to use (default: 20)
    
    Returns:
        str: LLM-generated response based on the relevant BNS document chunks
    """
    # Get relevant chunks
    chunks = query_bns(user_query, top_k)
    
    # Create context from chunks, with importance weighting
    context = ""
    for idx, chunk in enumerate(chunks):
        # Use relevance score to order chunks
        importance = top_k - idx  # Higher position = more important
        context += f"Document Chunk {idx+1} (Importance: {importance}/10):\n"
        
        # Add metadata if available
        section = chunk["metadata"].get("section", "")
        article = chunk["metadata"].get("article", "")
        chapter = chunk["metadata"].get("chapter", "")
        
        if section or article or chapter:
            metadata_info = []
            if section:
                metadata_info.append(f"Section {section}")
            if article:
                metadata_info.append(f"Article {article}")
            if chapter:
                metadata_info.append(f"Chapter {chapter}")
            
            context += f"[{', '.join(metadata_info)}]\n"
        
        context += f"{chunk['content']}\n\n"
    
    # Create system prompt with more detailed instructions
    system_prompt = """You are a legal assistant specialized in the Bharatiya Nyaya Sanhita (BNS), 
    which is the criminal code of India replacing the Indian Penal Code. Use the provided document 
    chunks to answer the user's question accurately and concisely.

    When answering:
    1. Cite specific sections and provisions from the BNS when relevant
    2. Explain legal concepts in clear, accessible language
    3. If the information is not present in the provided chunks, acknowledge this limitation
    4. Format your answer in a structured way for readability
    5. Pay special attention to chunks marked with higher importance ratings
    6. Include direct quotes from the BNS when appropriate, using quotation marks
    7. Distinguish between definitions, procedures, and penalties in your explanation

    Base your answer solely on the provided context. Do not make up information."""
    
    # Create user prompt with enhanced instructions
    user_prompt = f"""Based on the following chunks from the Bharatiya Nyaya Sanhita document, 
    please answer my question: "{user_query}"

    Context information:
    {context}

    Please provide a comprehensive but concise answer, citing specific sections, 
    and including direct quotes when appropriate."""
    
    # Get LLM response
    response = get_llm_response(system_prompt, user_prompt)
    return response


def query_judgments_with_llm(user_query, top_k=10):
    """
    Search for relevant judgment chunks and generate an LLM response
    
    Args:
        user_query (str): The user's question about legal judgments
        top_k (int): Number of most relevant chunks to use (default: 10)
    
    Returns:
        str: LLM-generated response based on the relevant judgment chunks
    """
    # Get relevant chunks
    chunks = query_judgments(user_query, top_k)
    
    # Create context from chunks with case hierarchy and importance
    context = ""
    # Group chunks by case title for better context organization
    cases = {}
    for chunk in chunks:
        title = chunk['title']
        if title not in cases:
            cases[title] = {
                'court': chunk['court'],
                'chunks': [],
                'url': chunk['url'],
                'year': chunk.get('year', '')
            }
        cases[title]['chunks'].append(chunk)
    
    # Sort cases by relevance (using the best chunk score from each case)
    for title, case_data in cases.items():
        case_data['best_score'] = min([c['relevance_score'] for c in case_data['chunks']])
    
    sorted_cases = sorted(cases.items(), key=lambda x: x[1]['best_score'])
    
    # Create context with hierarchical structure
    for idx, (title, case_data) in enumerate(sorted_cases):
        context += f"Case {idx+1}: {title} ({case_data['court']})\n"
        if case_data['year']:
            context += f"Year: {case_data['year']}\n"
        
        # Sort chunks within each case by position if available
        case_chunks = sorted(case_data['chunks'], 
                          key=lambda x: x.get('position', 0))
        
        for i, chunk in enumerate(case_chunks):
            context += f"Excerpt {i+1}:\n{chunk['content']}\n\n"
        
        context += "-" * 40 + "\n\n"
    
    # Create references section with citation information
    references = "Case References:\n"
    for idx, (title, case_data) in enumerate(sorted_cases):
        references += f"{idx+1}. {title} ({case_data['court']})"
        if case_data['year']:
            references += f", {case_data['year']}"
        references += f": {case_data['url']}\n"
    
    # Create system prompt with more detailed instructions
    system_prompt = """You are a legal assistant specialized in analyzing and explaining legal judgments 
    from Indian courts. Use the provided judgment excerpts to answer the user's question accurately.

    When answering:
    1. Cite specific cases and precedents when relevant, using proper citation format
    2. Explain the legal reasoning and principles established in the judgments
    3. Note any dissenting opinions or evolution of legal interpretation
    4. If the information is not present in the provided chunks, acknowledge this limitation
    5. Include case citations in standard legal format
    6. Consider the hierarchy of courts when analyzing precedent (Supreme Court > High Courts > Lower Courts)
    7. Mention the year of judgment when relevant to show legal evolution
    8. Use direct quotes from judgments when appropriate, with proper attribution

    Base your answer solely on the provided context. Do not make up information."""
    
    # Create user prompt with enhanced instructions
    user_prompt = f"""Based on the following excerpts from legal judgments, 
    please answer my question: "{user_query}"

    Context information:
    {context}

    {references}

    Please provide a comprehensive analysis based on these judgments, explaining the legal reasoning, 
    citing cases properly, and noting any evolution in legal interpretation."""
    
    # Get LLM response
    response = get_llm_response(system_prompt, user_prompt)
    return response

def query_constitution_with_llm(user_query, top_k=10):
    """
    Search for relevant constitution document chunks and generate an LLM response
    
    Args:
        user_query (str): The user's question about the Indian Constitution
        top_k (int): Number of most relevant chunks to use (default: 10)
    
    Returns:
        str: LLM-generated response based on the relevant constitution document chunks
    """
    # Get relevant chunks
    chunks = query_constitution(user_query, top_k)
    
    # Group chunks by articles, parts, and schedules for better context
    grouped_chunks = {}
    for chunk in chunks:
        # Get article/part/schedule info
        articles = chunk['metadata'].get('articles', [])
        part = chunk['metadata'].get('part', '')
        schedule = chunk['metadata'].get('schedule', '')
        
        # Create a key for grouping
        if articles:
            key = f"Article {', '.join(articles)}"
        elif part:
            key = f"Part {part}"
        elif schedule:
            key = f"Schedule {schedule}"
        else:
            page_num = chunk['metadata'].get('page', 'unknown')
            key = f"Page {page_num}"
            
        if key not in grouped_chunks:
            grouped_chunks[key] = []
            
        grouped_chunks[key].append(chunk)
    
    # Create context from grouped chunks
    context = ""
    section_count = 1
    
    for section_key, section_chunks in grouped_chunks.items():
        context += f"Section {section_count}: {section_key}\n"
        context += "-" * 40 + "\n"
        
        # Sort chunks by relevance within each section
        sorted_section_chunks = sorted(section_chunks, key=lambda x: x['relevance_score'])
        
        for i, chunk in enumerate(sorted_section_chunks):
            page_num = chunk['metadata'].get('page', 'unknown page')
            context += f"Excerpt {i+1} (Page {page_num}):\n{chunk['content']}\n\n"
            
        context += "\n"
        section_count += 1
    
    # Create system prompt with more detailed instructions
    system_prompt = """You are a constitutional law expert specializing in the Indian Constitution. 
    Use the provided excerpts to answer the user's question accurately and authoritatively.

    When answering:
    1. Cite specific Articles, Parts, Schedules, or Amendments when relevant
    2. Explain constitutional principles clearly, with their significance in India's legal framework
    3. Note any important legal interpretations by the Supreme Court when relevant
    4. If the information is not present in the provided chunks, acknowledge this limitation
    5. Structure your answer logically with appropriate headings if the answer is complex
    6. Explain the historical context of constitutional provisions when relevant
    7. Include direct quotes from the Constitution when appropriate
    8. Differentiate between Fundamental Rights, Directive Principles, and other constitutional elements

    Base your answer solely on the provided context. Do not make up information."""
    
    # Create user prompt with enhanced instructions
    user_prompt = f"""Based on the following excerpts from the Indian Constitution, 
    please answer my question: "{user_query}"

    Context information:
    {context}

    Please provide a comprehensive and accurate explanation based on the constitutional provisions,
    citing specific Articles, Parts or Schedules, and explaining their significance."""
    
    # Get LLM response
    response = get_llm_response(system_prompt, user_prompt)
    return response


print(query_bns_with_llm("A group attacks a person due to religion. Which provisions of bns handle this?"))
