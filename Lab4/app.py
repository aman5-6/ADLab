from flask import Flask, request, jsonify, render_template
import requests
import os
import fitz  # PyMuPDF for extracting text from PDFs

app = Flask(__name__)

# Set your Groq API key as an environment variable
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_ID = "llama-3.3-70b-versatile"

uploaded_file_content = ""  # Store extracted text from uploaded PDF

@app.route("/")
def home():
    """Serve the frontend HTML page."""
    return render_template("index.html")

def query_groq_llama(prompt, file_content=None):
    """Send a request to the Groq LLaMA API."""
    messages = [{"role": "user", "content": prompt}]
    if file_content:
        messages.append({"role": "system", "content": file_content})

    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": 500,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(GROQ_API_URL, json=payload, headers=headers)
    return response.json()

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests."""
    global uploaded_file_content
    data = request.json
    prompt = data.get("message", "")

    if not prompt:
        return jsonify({"error": "Message is required"}), 400

    response = query_groq_llama(prompt, uploaded_file_content)
    return jsonify(response)

@app.route("/upload", methods=["POST"])
def upload():
    """Handle PDF file uploads and extract text."""
    global uploaded_file_content

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    # Extract text from PDF
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])

    uploaded_file_content = text  # Store extracted text
    return jsonify({"message": "PDF uploaded successfully!", "extracted_text": text[:500] + "..."})  # Show preview

if __name__ == "__main__":
    app.run(debug=True)
