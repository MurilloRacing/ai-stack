import os
import tempfile
import io
import json
from typing import List, Dict, Any, Optional, Union

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

class RAGIndexer:
    """A class for indexing and querying documents using a RAG pipeline with Supabase storage."""
    
    def __init__(self, index_dir: str = "/root/flask_rag_api/chroma") -> None:
        """Initialize the RAGIndexer with storage and indexing configurations.

        Args:
            index_dir (str): Directory for ChromaDB persistence. Defaults to "/root/flask_rag_api/chroma".
        """
        self.index_dir: str = index_dir
        self.text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        self.embedder: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client: chromadb.Client = chromadb.Client(Settings(persist_directory=index_dir))
        self.collection: chromadb.api.models.Collection = self.chroma_client.get_or_create_collection("documents")
        self.redis_client: redis.Redis = redis.Redis(host='localhost', port=6379, db=0)
        self.supabase: Any = create_client(
            "https://pvwwcnxaogcdjswctnqn.supabase.co",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2d3djbnhhb2djZGpzd2N0bnFuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg0NTUzNzEsImV4cCI6MjA2NDAzMTM3MX0.LRnn_DYBDIrrR6NxZxsjy29vLIHqptxiL0XkFcB51kk"
        )
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[str] = []
        self.index_files()

    def read_file(self, file_path: str, file_data: Optional[bytes] = None) -> str:
        """Read and extract content from a file based on its extension.

        Args:
            file_path (str): Path to the file.
            file_data (Optional[bytes]): Optional byte data for direct reading (e.g., for Excel files).

        Returns:
            str: Extracted content from the file, or empty string if extraction fails.
        """
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
            return ""

    def index_files(self) -> None:
        """Index files from Supabase bucket and store embeddings in ChromaDB."""
        documents = []
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                response = self.supabase.storage.from_('company-files').list()
                if not response:
                    print("⚠️ No files found in Supabase bucket 'company-files'.")
                    return
            except Exception as e:
                print(f"❌ Error listing files in Supabase bucket: {e}")
                return

            for file_info in response:
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
                            metadata = {"file": filename, "path": f"company-files/{filename}", "chunk": i}
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
                    print(f"❌ Error indexing {file_path}: {e}")
                    continue

        self.corpus = documents
        if documents:
            tokenized_corpus = [doc.split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, str]]:
        """Query indexed documents using a combination of vector and BM25 search.

        Args:
            query_text (str): The query text to search for.
            top_k (int): Number of top results to return. Defaults to 3.

        Returns:
            List[Dict[str, str]]: List of dictionaries containing file, path, and content of matching documents.
        """
        cache_key = f"query:{query_text}"
        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)

        query_embedding = self.embedder.encode(query_text).tolist()
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        combined_results: Dict[str, float] = {}
        
        # Handle vector search results
        if vector_results['ids'] and vector_results['ids'][0]:
            print("📊 Vector Search Results:")
            for i, (doc_id, dist) in enumerate(zip(vector_results['ids'][0], vector_results['distances'][0])):
                file_name = vector_results['metadatas'][0][i]['file']
                content = vector_results['documents'][0][i]
                print(f"  - {doc_id} (file: {file_name}, distance: {dist:.4f}): {content[:50]}...")
                combined_results[doc_id] = combined_results.get(doc_id, 0) + (1 / (i + 1))

        # Handle BM25 search results
        if self.bm25 and self.corpus:
            tokenized_query = query_text.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_results = sorted(
                [(i, score) for i, score in enumerate(bm25_scores)],
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            print("📊 BM25 Search Results:")
            for rank, (idx, score) in enumerate(bm25_results):
                try:
                    # Generate document ID using corpus content
                    content = self.corpus[idx]
                    # Look for matching content in vector results
                    for i, doc_content in enumerate(vector_results['documents'][0]):
                        if content == doc_content:
                            doc_id = vector_results['ids'][0][i]
                            print(f"  - {doc_id} (score: {score:.4f}): {content[:50]}...")
                            combined_results[doc_id] = combined_results.get(doc_id, 0) + (1 / (rank + 1))
                            break
                except (IndexError, KeyError) as e:
                    print(f"⚠️ Error processing BM25 result at index {idx}: {str(e)}")
                    continue

        if not combined_results:
            print("⚠️ No results found for query.")
            return []

        # Process final results
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        print("📊 Final Combined Results (Top 3):")
        for doc_id, score in sorted_results:
            print(f"  - {doc_id} (combined score: {score:.4f})")

        final_results: List[Dict[str, str]] = []
        for doc_id, _ in sorted_results:
            try:
                idx = vector_results['ids'][0].index(doc_id)
                final_results.append({
                    "file": vector_results['metadatas'][0][idx]['file'],
                    "path": vector_results['metadatas'][0][idx]['path'],
                    "content": vector_results['documents'][0][idx]
                })
            except (ValueError, IndexError) as e:
                print(f"⚠️ Error retrieving results for doc_id {doc_id}: {str(e)}")
                continue

        self.redis_client.setex(cache_key, 3600, json.dumps(final_results))
        return final_results

if __name__ == "__main__":
    indexer = RAGIndexer()
    results = indexer.query("What automations use binary_sensor.kitchen_motion?")
    print(results)
