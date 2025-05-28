import os
import tempfile
import PyPDF2
import yaml
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings
import redis
import json
import markdown
from docx import Document
from PIL import Image
import pytesseract
from lxml import etree
from bs4 import BeautifulSoup
from supabase import create_client, Client

class RAGIndexer:
    def __init__(self, index_dir="/root/flask_rag_api/chroma"):
        self.index_dir = index_dir
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.Client(Settings(persist_directory=index_dir))
        self.collection = self.chroma_client.get_or_create_collection("documents")
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.supabase = create_client(
            "https://pvwwcnxaogcdjswctnqn.supabase.co",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2d3djbnhhb2djZGpzd2N0bnFuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg0NTUzNzEsImV4cCI6MjA2NDAzMTM3MX0.LRnn_DYBDIrrR6NxZxsjy29vLIHqptxiL0XkFcB51kk"
        )
        self.bm25 = None
        self.corpus = []
        self.index_files()

    def read_file(self, file_path):
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
                df = pd.read_excel(file_path)
                return df.to_string()
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
            print(f"Error reading {file_path}: {e}")
            return ""

    def index_files(self):
        documents = []
        with tempfile.TemporaryDirectory() as temp_dir:
            # List files in Supabase bucket
            response = self.supabase.storage.from_('company-files').list()
            for file_info in response:
                filename = file_info['name']
                # Download file to temp directory
                file_path = os.path.join(temp_dir, filename)
                file_data = self.supabase.storage.from_('company-files').download(filename)
                with open(file_path, 'wb') as f:
                    f.write(file_data)
                try:
                    content = self.read_file(file_path)
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
                except Exception as e:
                    print(f"Error indexing {file_path}: {e}")
        self.corpus = documents
        # Only initialize BM25Okapi if there are documents
        if documents:
            tokenized_corpus = [doc.split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None  # Set to None if no documents

    def query(self, query_text, top_k=3):
        cache_key = f"query:{query_text}"
        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)

        query_embedding = self.embedder.encode(query_text).tolist()
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        combined_results = {}
        # Use vector search results
        if vector_results['ids'][0]:  # Check if there are any vector results
            for i, (doc_id, dist) in enumerate(zip(vector_results['ids'][0], vector_results['distances'][0])):
                combined_results[doc_id] = combined_results.get(doc_id, 0) + (1 / (i + 1))

        # Use BM25 search results if available
        if self.bm25 and self.corpus:
            tokenized_query = query_text.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_results = sorted(
                [(i, score) for i, score in enumerate(bm25_scores)],
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            for rank, (idx, score) in enumerate(bm25_results):
                doc_id = f"{vector_results['metadatas'][0][idx]['file']}_{idx}"
                combined_results[doc_id] = combined_results.get(doc_id, 0) + (1 / (rank + 1))

        if not combined_results:
            return []  # Return empty list if no results

        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        final_results = []
        for doc_id, _ in sorted_results:
            for i, d_id in enumerate(vector_results['ids'][0]):
                if d_id == doc_id:
                    final_results.append({
                        "file": vector_results['metadatas'][0][i]['file'],
                        "path": vector_results['metadatas'][0][i]['path'],
                        "content": vector_results['documents'][0][i]
                    })
                    break

        self.redis_client.setex(cache_key, 3600, json.dumps(final_results))
        return final_results

if __name__ == "__main__":
    indexer = RAGIndexer()
    results = indexer.query("What automations use binary_sensor.kitchen_motion?")
    print(results)
