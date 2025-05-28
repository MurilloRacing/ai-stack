import os
import mimetypes
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_httpauth import HTTPBasicAuth
import logging
from logging.handlers import RotatingFileHandler
from supabase import create_client, Client
from rag_indexer import RAGIndexer

app = Flask(__name__)

limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])
auth = HTTPBasicAuth()
users = {"admin": "your_secure_password"}

@auth.verify_password
def verify_password(username, password):
    if username in users and users[username] == password:
        return username
    return None

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

@app.route('/upload', methods=['POST'])
@auth.login_required
@limiter.limit("10 per minute")
def upload_file():
    if 'file' not in request.files:
        app.logger.error("No file part in request")
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        app.logger.error("No selected file")
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        try:
            # Read the file contents into bytes
            file_content = file.read()
            # Determine the MIME type based on the file extension
            mime_type, _ = mimetypes.guess_type(file.filename)
            if not mime_type:
                # Fallback for .xlsx if mimetypes fails
                mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' if file.filename.endswith('.xlsx') else 'application/octet-stream'
            file_options = {
                'content-type': mime_type,
                'upsert': 'true'
            }
            response = supabase.storage.from_('company-files').upload(
                file.filename, file_content, file_options=file_options
            )
            app.logger.info(f"File uploaded to Supabase: {file.filename} by user {auth.current_user()}")
            return jsonify({"message": f"File {file.filename} uploaded successfully"}), 200
        except Exception as e:
            app.logger.error(f"Error uploading file to Supabase: {str(e)}")
            return jsonify({"error": str(e)}), 500
    app.logger.error(f"Invalid file type: {file.filename}")
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/rag_query', methods=['POST'])
@auth.login_required
@limiter.limit("20 per minute")
def rag_query():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        app.logger.error("No query provided")
        return jsonify({"answer": "No query provided."}), 400
    app.logger.info(f"Query received: {query} by user {auth.current_user()}")
    results = rag_indexer.query(query, top_k=3)
    if not results:
        return jsonify({"answer": "No relevant data found in private directories."})
    answer = "Found relevant data:\n"
    for result in results:
        answer += f"- {result['file']} ({result['path']}):\n{result['content']}\n\n"
    return jsonify({"answer": answer.strip()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
