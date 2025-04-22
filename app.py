import os
import json
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from huggingface_api import HuggingFaceAPI
from utils import base64_to_image

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
DB_PATH = os.environ.get("DB_PATH", "./images.db")
MODEL_ID = os.environ.get("HF_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

# Initialize database
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            image_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            seed INTEGER,
            params TEXT,
            feedback INTEGER DEFAULT 0
        )
        ''')
        conn.commit()

init_db()

# Initialize the Hugging Face API service
hf_api = HuggingFaceAPI(model_id=MODEL_ID, api_token=HF_API_TOKEN)

@app.route('/api/generate', methods=['POST'])
def generate_image():
    data = request.json
    
    if not data or 'prompt' not in data:
        return jsonify({"error": "Prompt is required"}), 400
    
    prompt = data.get('prompt')
    negative_prompt = data.get('negative_prompt', '')
    height = int(data.get('height', 512))
    width = int(data.get('width', 512))
    num_inference_steps = int(data.get('num_inference_steps', 50))
    guidance_scale = float(data.get('guidance_scale', 7.5))
    seed = data.get('seed')
    
    # Validate dimensions
    if height < 128 or height > 1024 or width < 128 or width > 1024:
        return jsonify({"error": "Height and width must be between 128 and 1024"}), 400
    
    # Validate inference steps
    if num_inference_steps < 1 or num_inference_steps > 150:
        return jsonify({"error": "Number of inference steps must be between 1 and 150"}), 400
    
    # Validate guidance scale
    if guidance_scale < 1.0 or guidance_scale > 20.0:
        return jsonify({"error": "Guidance scale must be between 1.0 and 20.0"}), 400
    
    try:
        # Generate image using Hugging Face API
        result = hf_api.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        
        # Save to database
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            params = json.dumps({
                'negative_prompt': negative_prompt,
                'height': height,
                'width': width,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
            })
            
            cursor.execute(
                'INSERT INTO images (prompt, image_data, seed, params) VALUES (?, ?, ?, ?)',
                (prompt, result['image'], result['seed'], params)
            )
            
            image_id = cursor.lastrowid
            conn.commit()
        
        return jsonify({
            "id": image_id,
            "image": result['image'],
            "prompt": result['prompt'],
            "seed": result['seed']
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/variations', methods=['POST'])
def generate_variations():
    data = request.json
    
    if not data or 'image' not in data:
        return jsonify({"error": "Image is required"}), 400
    
    image_data = data.get('image')
    prompt = data.get('prompt', '')
    negative_prompt = data.get('negative_prompt', '')
    strength = float(data.get('strength', 0.75))
    num_inference_steps = int(data.get('num_inference_steps', 50))
    guidance_scale = float(data.get('guidance_scale', 7.5))
    num_variations = int(data.get('num_variations', 4))
    
    # Validate parameters
    if strength < 0.0 or strength > 1.0:
        return jsonify({"error": "Strength must be between 0.0 and 1.0"}), 400
    
    if num_variations < 1 or num_variations > 10:
        return jsonify({"error": "Number of variations must be between 1 and 10"}), 400
    
    try:
        # Convert base64 to PIL Image
        image = base64_to_image(image_data)
        
        # Generate variations using Hugging Face API
        results = hf_api.generate_variations(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_variations=num_variations,
        )
        
        variations = []
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            for result in results:
                params = json.dumps({
                    'negative_prompt': negative_prompt,
                    'strength': strength,
                    'num_inference_steps': num_inference_steps,
                    'guidance_scale': guidance_scale,
                })
                
                cursor.execute(
                    'INSERT INTO images (prompt, image_data, seed, params) VALUES (?, ?, ?, ?)',
                    (result['prompt'], result['image'], result['seed'], params)
                )
                
                variation_id = cursor.lastrowid
                
                variations.append({
                    "id": variation_id,
                    "image": result['image'],
                    "prompt": result['prompt'],
                    "seed": result['seed']
                })
            
            conn.commit()
        
        return jsonify(variations)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/images', methods=['GET'])
def get_images():
    try:
        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Validate parameters
        if limit < 1 or limit > 100:
            return jsonify({"error": "Limit must be between 1 and 100"}), 400
        
        if offset < 0:
            return jsonify({"error": "Offset must be non-negative"}), 400
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT id, prompt, image_data, created_at, seed, feedback FROM images ORDER BY created_at DESC LIMIT ? OFFSET ?',
                (limit, offset)
            )
            
            images = [dict(row) for row in cursor.fetchall()]
        
        return jsonify(images)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/images/<int:image_id>', methods=['GET'])
def get_image(image_id):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM images WHERE id = ?', (image_id,))
            
            row = cursor.fetchone()
        
        if not row:
            return jsonify({"error": "Image not found"}), 404
        
        return jsonify(dict(row))
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def save_feedback():
    data = request.json
    
    if not data or 'image_id' not in data or 'feedback' not in data:
        return jsonify({"error": "Image ID and feedback are required"}), 400
    
    image_id = data.get('image_id')
    feedback = data.get('feedback')
    
    # Validate feedback value
    if not isinstance(feedback, int) or feedback < -1 or feedback > 1:
        return jsonify({"error": "Feedback must be -1, 0, or 1"}), 400
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            cursor.execute('UPDATE images SET feedback = ? WHERE id = ?', (feedback, image_id))
            
            if cursor.rowcount == 0:
                return jsonify({"error": "Image not found"}), 404
            
            conn.commit()
        
        return jsonify({"success": True})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)