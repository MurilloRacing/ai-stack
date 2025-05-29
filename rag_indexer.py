import os
import tempfile
import io
import json
from typing import List, Dict, Any, Optional, Union, Tuple

# Third-party imports
import PyPDF2
import yaml
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings
import redis
import markdown
from docx import Document
from PIL import Image
import pytesseract
from lxml import etree
from bs4 import BeautifulSoup
from supabase import create_client
from tenacity import retry, stop_after_attempt, wait_exponential

class RAGIndexer:
    """A class for indexing and querying documents using a RAG pipeline with Supabase storage.
    
    This class provides functionality to:
    - Read and process various file types (PDF, XLSX, YAML, etc.)
    - Store document embeddings in ChromaDB
    - Cache query results in Redis
    - Combine vector and BM25 search for better results
    
    Attributes:
        index_dir (str): Directory for ChromaDB persistence
        text_splitter (RecursiveCharacterTextSplitter): Text chunking utility
        embedder (SentenceTransformer): Document embedding model
        chroma_client (chromadb.Client): ChromaDB client
        collection (chromadb.Collection): ChromaDB collection
        redis_client (redis.Redis): Redis client for caching
        supabase (Client): Supabase client
        bm25 (Optional[BM25Okapi]): BM25 search index
        corpus (List[str]): List of indexed documents
    """
    
    def __init__(self, index_dir: str = "/root/flask_rag_api/chroma") -> None:
        """Initialize the RAGIndexer with storage and indexing configurations.

        Args:
            index_dir (str): Directory for ChromaDB persistence. Defaults to "/root/flask_rag_api/chroma".
        
        Raises:
            ValueError: If index_dir is invalid
            ConnectionError: If Redis or Supabase connection fails
        """
        try:
            # Validate index directory
            if not index_dir:
                raise ValueError("index_dir cannot be empty")
            self.index_dir: str = index_dir
            os.makedirs(index_dir, exist_ok=True)

            # Initialize text splitter
            self.text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
                chunk_size=500, 
                chunk_overlap=50
            )

            # Initialize embedder
            try:
                self.embedder: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                raise RuntimeError(f"Failed to initialize SentenceTransformer: {str(e)}")

            # Initialize ChromaDB
            try:
                self.chroma_client: chromadb.Client = chromadb.Client(Settings(persist_directory=index_dir))
                self.collection: chromadb.api.models.Collection = self.chroma_client.get_or_create_collection("documents")
            except Exception as e:
                raise ConnectionError(f"Failed to initialize ChromaDB: {str(e)}")

            # Initialize Redis
            try:
                self.redis_client: redis.Redis = redis.Redis(
                    host='localhost', 
                    port=6379, 
                    db=0, 
                    socket_timeout=5
                )
                self.redis_client.ping()  # Test connection
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Redis: {str(e)}")

            # Initialize Supabase
            try:
                self.supabase: Any = create_client(
                    "https://pvwwcnxaogcdjswctnqn.supabase.co",
                    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2d3djbnhhb2djZGpzd2N0bnFuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg0NTUzNzEsImV4cCI6MjA2NDAzMTM3MX0.LRnn_DYBDIrrR6NxZxsjy29vLIHqptxiL0XkFcB51kk"
                )
            except Exception as e:
                raise ConnectionError(f"Failed to initialize Supabase client: {str(e)}")

            self.bm25: Optional[BM25Okapi] = None
            self.corpus: List[str] = []
            
            # Initial indexing
            try:
                self.index_files()
            except Exception as e:
                print(f"⚠️ Initial indexing failed: {str(e)}")
        
        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def read_file(self, file_path: str, file_data: Optional[bytes] = None) -> str:
        """Read and extract content from a file based on its extension.

        Args:
            file_path (str): Path to the file.
            file_data (Optional[bytes]): Optional byte data for direct reading.

        Returns:
            str: Extracted content from the file.

        Raises:
            ValueError: If file_path is invalid
            IOError: If file reading fails
        """
        if not file_path:
            raise ValueError("file_path cannot be empty")

        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.pdf':
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    return " ".join(page.extract_text() for page in pdf.pages)
            elif ext in ('.yaml', '.yml'):
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
                    return json.dumps(data)
            elif ext == '.xlsx':
                try:
                    # Try reading directly from the byte stream if provided
                    if file_data is not None:
                        print(f"📄 Attempting to read {file_path} directly from byte stream...")
                        df = pd.read_excel(io.BytesIO(file_data), engine="openpyxl")
                    else:
                        print(f"📄 Attempting to read {file_path} from disk...")
                        df = pd.read_excel(file_path, engine="openpyxl")
                    # Convert DataFrame to a cleaner string format for search
                    content = " ".join(f"{col}: {val}" for _, row in df.iterrows() for col, val in row.items())
                    print(f"✅ Successfully read Excel file {file_path}. Content:\n{content}")
                    return content
                except Exception as e:
                    print(f"❌ Failed to read Excel file {file_path}: {str(e)}")
                    return ""
            elif ext == '.md':
                with open(file_path, 'r') as f:
                    md_content = f.read()
                    return markdown.markdown(md_content)
            elif ext == '.docx':
                doc = Document(file_path)
                return " ".join(paragraph.text for paragraph in doc.paragraphs)
            elif ext in ('.png', '.jpg', '.jpeg'):
                image = Image.open(file_path)
                return pytesseract.image_to_string(image)
            elif ext == '.xml':
                tree = etree.parse(file_path)
                return etree.tostring(tree, pretty_print=True, encoding='unicode')
            elif ext == '.html':
                with open(file_path, 'r') as f:
                    soup = BeautifulSoup(f, 'html.parser')
                    return soup.get_text()
            else:
                with open(file_path, 'r') as f:
                    return f.read()
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            raise IOError(f"Failed to read file: {str(e)}")

    def index_files(self) -> None:
        """Index files from Supabase bucket and store embeddings in ChromaDB.
        
        Raises:
            ConnectionError: If Supabase or ChromaDB operations fail
        """
        documents: List[str] = []
        temp_dir = None
        
        try:
            temp_dir = tempfile.mkdtemp()
            response = self.supabase.storage.from_('company-files').list()
            
            if not response:
                print("⚠️ No files found in Supabase bucket 'company-files'.")
                return

            for file_info in response:
                try:
                    filename = file_info['name']
                    print(f"📄 Processing file: {filename}")
                    file_path = os.path.join(temp_dir, filename)
                    
                    try:
                        file_data = self.supabase.storage.from_('company-files').download(filename)
                        print(f"📥 Downloaded {filename}, size: {len(file_data)} bytes")
                        
                        with open(file_path, 'wb') as f:
                            f.write(file_data)
                        print(f"💾 Wrote {filename} to temporary path: {file_path}")
                        
                        if filename.endswith('.xlsx'):
                            content = self.read_file(file_path, file_data=file_data)
                        else:
                            content = self.read_file(file_path)
                            
                        if content:
                            chunks = self.text_splitter.split_text(content)
                            for i, chunk in enumerate(chunks):
                                doc_id = f"{filename}_{i}"
                                embedding = self.embedder.encode(chunk).tolist()
                                metadata = {
                                    "file": filename, 
                                    "path": f"company-files/{filename}", 
                                    "chunk": i
                                }
                                self.collection.add(
                                    ids=[doc_id],
                                    documents=[chunk],
                                    embeddings=[embedding],
                                    metadatas=[metadata]
                                )
                                documents.append(chunk)
                        else:
                            print(f"⚠️ Skipping {filename}: No content extracted.")
                            
                    except Exception as e:
                        print(f"❌ Error processing {filename}: {str(e)}")
                        continue
                        
                except Exception as e:
                    print(f"❌ Error with file info: {str(e)}")
                    continue
                
        # Update corpus and BM25 index
        self.corpus = documents
        if documents:
            tokenized_corpus = [doc.split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None
            
    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, str]]:
        """Query indexed documents using vector and BM25 search.

        Args:
            query_text (str): The query text to search for.
            top_k (int): Number of top results to return.

        Returns:
            List[Dict[str, str]]: List of matched documents.

        Raises:
            ValueError: If query parameters are invalid
        """
        # Validate input parameters
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty")
        
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        # Check cache first
        cache_key = f"query:{query_text}"
        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)

        # Get vector search results
        try:
            query_embedding = self.embedder.encode(query_text).tolist()
            vector_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
        except Exception as e:
            print(f"❌ Error performing vector search: {e}")
            return []

        # Validate vector results structure
        if not vector_results or not isinstance(vector_results, dict):
            print("⚠️ Invalid vector results structure")
            return []

        required_keys = ['ids', 'distances', 'metadatas', 'documents']
        if not all(key in vector_results for key in required_keys):
            print("⚠️ Missing required keys in vector results")
            return []

        if not all(isinstance(vector_results[key], list) and len(vector_results[key]) > 0 for key in required_keys):
            print("⚠️ Empty or invalid vector results arrays")
            return []

        combined_results: Dict[str, float] = {}
        
        # Process vector search results
        print("📊 Vector Search Results:")
        for i in range(len(vector_results['ids'][0])):
            try:
                doc_id = vector_results['ids'][0][i]
                dist = vector_results['distances'][0][i]
                metadata = vector_results['metadatas'][0][i]
                content = vector_results['documents'][0][i]
                
                if not isinstance(metadata, dict) or 'file' not in metadata:
                    print(f"⚠️ Invalid metadata for document {doc_id}")
                    continue
                    
                file_name = metadata['file']
                print(f"  - {doc_id} (file: {file_name}, distance: {dist:.4f}): {content[:50]}...")
                combined_results[doc_id] = combined_results.get(doc_id, 0) + (1 / (i + 1))
            except (IndexError, KeyError) as e:
                print(f"⚠️ Error processing vector result at index {i}: {str(e)}")
                continue

        # Process BM25 search results
        if self.bm25 and self.corpus:
            try:
                tokenized_query = query_text.split()
                bm25_scores = self.bm25.get_scores(tokenized_query)
                
                # Get top BM25 results
                bm25_results = sorted(
                    [(i, score) for i, score in enumerate(bm25_scores) if score > 0],
                    key=lambda x: x[1],
                    reverse=True
                )[:top_k]
                
                if bm25_results:
                    print("📊 BM25 Search Results:")
                    for rank, (idx, score) in enumerate(bm25_results):
                        if idx >= len(self.corpus):
                            continue
                            
                        content = self.corpus[idx]
                        # Find matching document in vector results
                        for i, vec_content in enumerate(vector_results['documents'][0]):
                            if content == vec_content:
                                doc_id = vector_results['ids'][0][i]
                                print(f"  - {doc_id} (score: {score:.4f}): {content[:50]}...")
                                combined_results[doc_id] = combined_results.get(doc_id, 0) + (1 / (rank + 1))
                                break
            except Exception as e:
                print(f"⚠️ Error in BM25 search: {str(e)}")

        if not combined_results:
            print("⚠️ No results found for query")
            return []

        # Build final results
        final_results: List[Dict[str, str]] = []
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        print("📊 Final Combined Results:")
        for doc_id, score in sorted_results:
            try:
                idx = vector_results['ids'][0].index(doc_id)
                metadata = vector_results['metadatas'][0][idx]
                content = vector_results['documents'][0][idx]
                
                if not isinstance(metadata, dict) or not all(k in metadata for k in ['file', 'path']):
                    print(f"⚠️ Invalid metadata structure for {doc_id}")
                    continue
                    
                result = {
                    "file": metadata['file'],
                    "path": metadata['path'],
                    "content": content
                }
                final_results.append(result)
                print(f"  - {doc_id} (score: {score:.4f}): {content[:50]}...")
            except (ValueError, IndexError, KeyError) as e:
                print(f"⚠️ Error processing final result for {doc_id}: {str(e)}")
                continue

        if final_results:
            self.redis_client.setex(cache_key, 3600, json.dumps(final_results))
            
        return final_results

    def cleanup(self) -> None:
        """Cleanup resources and close connections."""
        try:
            self.redis_client.close()
            print("✅ Redis connection closed")
        except Exception as e:
            print(f"⚠️ Error closing Redis connection: {str(e)}")
    
        try:
            self.chroma_client.close()
            print("✅ ChromaDB connection closed")
        except Exception as e:
            print(f"⚠️ Error closing ChromaDB connection: {str(e)}")

if __name__ == "__main__":
    indexer = RAGIndexer()
    results = indexer.query("What automations use binary_sensor.kitchen_motion?")
    print(results)
