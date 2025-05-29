import os
import mimetypes
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from limits.storage import RedisStorage  # Changed import
from flask_httpauth import HTTPBasicAuth
import logging
from logging.handlers import RotatingFileHandler
from supabase import create_client, Client
from rag_indexer import RAGIndexer

app = Flask(__name__)

# Configure Redis storage for rate limiting
try:
    redis_storage = RedisStorage('redis://localhost:6379', ssl_cert_reqs=None)
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        storage=redis_storage,
        default_limits=["200 per day", "50 per hour"]
    )
    app.logger.info("✅ Redis rate limiting configured successfully")
except Exception as e:
    app.logger.error(f"❌ Redis configuration failed: {str(e)}")
    # Fallback to default storage
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"]
    )
    app.logger.warning("⚠️ Using default storage for rate limiting")

# 4. Add proper application configuration
app.config.update(
    JSON_SORT_KEYS=False,
    MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16MB max file size
)

auth = HTTPBasicAuth()
users = {"admin": "your_secure_password"}

@auth.verify_password
def verify_password(username, password):
    if username in users and users[username] == password:
        return username
    return None

# Configure logging
if not os.path.exists('logs'):
    os.mkdir('logs')
handler = RotatingFileHandler('logs/api.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

# Supabase configuration
SUPABASE_URL = "https://pvwwcnxaogcdjswctnqn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2d3djbnhhb2djZGpzd2N0bnFuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg0NTUzNzEsImV4cCI6MjA2NDAzMTM3MX0.LRnn_DYBDIrrR6NxZxsjy29vLIHqptxiL0XkFcB51kk"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'yaml', 'yml', 'xlsx', 'md', 'png', 'jpg', 'jpeg', 'xml', 'html'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

rag_indexer = RAGIndexer()

# 3. Add better error handling for file uploads
@app.route('/upload', methods=['POST'])
@auth.login_required
@limiter.limit("10 per minute")
def upload_file():
    try:
        if 'file' not in request.files:
            app.logger.error("No file part in request")
            return jsonify({"error": "No file part"}), 400
            
        file = request.files['file']
        if file.filename == '':
            app.logger.error("No selected file")
            return jsonify({"error": "No selected file"}), 400
            
        if not allowed_file(file.filename):
            app.logger.error(f"Invalid file type: {file.filename}")
            return jsonify({"error": "File type not allowed"}), 400

        file_content = file.read()
        mime_type, _ = mimetypes.guess_type(file.filename)
        
        if not mime_type:
            mime_type = ('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' 
                        if file.filename.endswith('.xlsx') else 'application/octet-stream')
            
        file_options = {
            'content-type': mime_type,
            'upsert': 'true'
        }
        
        response = supabase.storage.from_('company-files').upload(
            file.filename, file_content, file_options=file_options
        )
        
        app.logger.info(f"File uploaded successfully: {file.filename} by user {auth.current_user()}")
        return jsonify({"message": f"File {file.filename} uploaded successfully"}), 200
        
    except Exception as e:
        app.logger.error(f"Error uploading file: {str(e)}")
        return jsonify({"error": "Error uploading file"}), 500

# 2. Add better error handling for RAG query endpoint
@app.route('/rag_query', methods=['POST'])
@auth.login_required
@limiter.limit("20 per minute")
def rag_query():
    try:
        data = request.get_json()
        if not data:
            app.logger.error("Invalid JSON data")
            return jsonify({"error": "Invalid JSON data"}), 400
            
        query = data.get('query', '').strip()
        if not query:
            app.logger.error("No query provided")
            return jsonify({"error": "Query cannot be empty"}), 400

        app.logger.info(f"Processing query: {query} by user {auth.current_user()}")
        results = rag_indexer.query(query, top_k=3)
        
        if not results:
            app.logger.info(f"No results found for query: {query}")
            return jsonify({"answer": "No relevant data found in private directories."})
            
        answer = "Found relevant data:\n"
        for result in results:
            answer += f"- {result['file']} ({result['path']}):\n{result['content']}\n\n"
            
        return jsonify({"answer": answer.strip()})
        
    except Exception as e:
        app.logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# 5. Add health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
