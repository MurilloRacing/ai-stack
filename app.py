from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/rag_query', methods=['POST'])
def rag_query():
    data = request.get_json()
    query = data.get('query', '')
    # Replace with your actual RAG logic; this is a placeholder
    response = {"answer": f"Response to: {query}"}
    return jsonify(response)

@app.route('/openapi.json', methods=['GET'])
def openapi():
    # Placeholder OpenAPI spec
    return jsonify({
        "openapi": "3.0.0",
        "info": {"title": "RAG API", "version": "1.0.0"},
        "paths": {
            "/rag_query": {
                "post": {
                    "summary": "Query the RAG API",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"type": "object", "properties": {"query": {"type": "string"}}}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}
                                }
                            }
                        }
                    }
                }
            }
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
