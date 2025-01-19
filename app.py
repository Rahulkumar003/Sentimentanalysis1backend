from flask import Flask, request, jsonify
from textblob import TextBlob
from flask_cors import CORS
import os
import requests
from typing import List, Dict
import numpy as np

app = Flask(__name__)
CORS(app)

# Hugging Face API configuration
HF_API_TOKEN = os.getenv('HF_API_TOKEN')  # Make sure to set this in your environment variables
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def query_huggingface_api(texts: List[str]) -> List[Dict]:
    """Send request to Hugging Face API and return predictions."""
    try:
        response = requests.post(API_URL, headers=HEADERS, json={"inputs": texts})
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")

def chunk_text(text: str, max_length: int = 1000) -> List[str]:
    """Split text into chunks that respect sentence boundaries."""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) <= max_length:
            current_chunk.append(sentence)
            current_length += len(sentence)
        else:
            if current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = len(sentence)
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks

@app.route('/')
def home():
    return jsonify({"status": "healthy", "message": "Sentiment Analysis API is running"})

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    """Endpoint to analyze sentiment."""
    try:
        data = request.get_json()
        if 'text' not in data:
            return jsonify({'error': 'Text not provided'}), 400

        text = data['text']
        method = data.get('method', 'textblob').lower()

        if method == 'textblob':
            # TextBlob analysis remains unchanged as it's lightweight
            analysis = TextBlob(text)
            sentiment = analysis.sentiment
            result = {
                'method': 'textblob',
                'polarity': sentiment.polarity,
                'subjectivity': sentiment.subjectivity,
                'sentiment': (
                    'Positive' if sentiment.polarity > 0 else
                    'Negative' if sentiment.polarity < 0 else
                    'Neutral'
                )
            }

        elif method == 'transformers':
            # Split text into manageable chunks
            chunks = chunk_text(text)
            
            # Get predictions from Hugging Face API
            predictions = query_huggingface_api(chunks)
            
            # Process results
            chunk_results = []
            for pred in predictions:
                # Extract the sentiment with highest score
                sentiment = max(pred, key=lambda x: x['score'])
                chunk_results.append({
                    'label': sentiment['label'],
                    'score': sentiment['score']
                })
            
            # Calculate statistics
            positive_chunks = sum(1 for r in chunk_results if r['label'] == 'POSITIVE')
            negative_chunks = sum(1 for r in chunk_results if r['label'] == 'NEGATIVE')
            
            avg_positive_score = np.mean([r['score'] for r in chunk_results if r['label'] == 'POSITIVE']) if positive_chunks > 0 else 0
            avg_negative_score = np.mean([r['score'] for r in chunk_results if r['label'] == 'NEGATIVE']) if negative_chunks > 0 else 0
            
            result = {
                'method': 'transformers',
                'positive_chunks': positive_chunks,
                'negative_chunks': negative_chunks,
                'total_chunks': len(chunk_results),
                'avg_positive_score': float(avg_positive_score),
                'avg_negative_score': float(avg_negative_score),
                'overall_sentiment': 'Positive' if positive_chunks > negative_chunks else 'Negative',
                'chunk_details': chunk_results
            }

        else:
            return jsonify({'error': 'Invalid method specified. Use "textblob" or "transformers".'}), 400

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
