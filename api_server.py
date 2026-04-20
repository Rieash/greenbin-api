# ============================================
# GREENBIN WASTE CLASSIFICATION API
# Photo comparison for ESP32-CAM
# ============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ==========================================
# MOBILE APP - CLASSIFICATION HISTORY
# ==========================================
classification_history = []
MAX_HISTORY = 100  # Keep last 100 items

def add_to_history(material, detailed_type, confidence, weight=0):
    """Store classification result for mobile app"""
    entry = {
        'id': len(classification_history) + 1,
        'timestamp': datetime.now().isoformat(),
        'material': material,
        'detailed_type': detailed_type,
        'confidence': confidence,
        'weight': weight,
        'sorted': True
    }
    classification_history.insert(0, entry)  # Add to beginning
    # Keep only last 100
    if len(classification_history) > MAX_HISTORY:
        classification_history.pop()

# ==========================================
# LOAD YOUR REFERENCE PHOTOS FROM D: DRIVE
# ==========================================
def load_references():
    refs = {
        'black_plastic': [],
        'white_paper': [],
        'clear_plastic': []
    }

    # Look for references in current directory (works on Windows & Linux)
    base_path = os.path.join(os.path.dirname(__file__), 'references')

    for material in refs.keys():
        folder = os.path.join(base_path, material)
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG')):
                    path = os.path.join(folder, filename)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (96, 96))
                        refs[material].append(img)
                        print(f"Loaded: {material}/{filename}")
        else:
            print(f"Warning: Folder not found: {folder}")

    return refs

print("=" * 50)
print("GREENBIN WASTE CLASSIFICATION API")
print("=" * 50)
print(f"Loading reference photos from {os.path.join(os.path.dirname(__file__), 'references')}...")

references = load_references()

print(f"\nBlack plastic: {len(references['black_plastic'])} photos")
print(f"White paper:   {len(references['white_paper'])} photos")
print(f"Clear plastic: {len(references['clear_plastic'])} photos")
print("\n" + "=" * 50)
print("API Ready!")
print("=" * 50 + "\n")

# ==========================================
# COMPARE IMAGES - FIND BEST MATCH
# ==========================================
def compare_images(img1, img2):
    # Resize to standard size
    img1 = cv2.resize(img1, (96, 96))
    img2 = cv2.resize(img2, (96, 96))
    
    # Method 1: Histogram comparison (brightness/texture)
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # Method 2: Structural similarity
    diff = cv2.absdiff(img1, img2)
    pixel_score = 1 - (np.mean(diff) / 255.0)
    
    # Method 3: Edge/shape comparison
    edges1 = cv2.Canny(img1, 50, 150)
    edges2 = cv2.Canny(img2, 50, 150)
    edge_diff = cv2.absdiff(edges1, edges2)
    edge_score = 1 - (np.mean(edge_diff) / 255.0)
    
    # Combined score
    final_score = (hist_score * 0.5) + (pixel_score * 0.3) + (edge_score * 0.2)
    return final_score

# ==========================================
# API ENDPOINT - ESP32 SENDS PHOTO HERE
# ==========================================
@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Get image from ESP32
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        nparr = np.frombuffer(file.read(), np.uint8)
        new_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if new_img is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        print(f"\n[NEW REQUEST] Image size: {new_img.shape}")
        
        # Compare to all reference photos
        scores = {
            'black_plastic': [],
            'white_paper': [],
            'clear_plastic': []
        }
        
        for material, ref_list in references.items():
            for ref_img in ref_list:
                score = compare_images(new_img, ref_img)
                scores[material].append(score)
        
        # Get best match from each category
        best_scores = {
            material: max(scores[material]) if scores[material] else 0
            for material in scores.keys()
        }
        
        # Determine winner
        detected_material = max(best_scores, key=best_scores.get)
        confidence = best_scores[detected_material]
        
        # Map to simple paper/plastic for Arduino
        result_map = {
            'black_plastic': 'plastic',
            'white_paper': 'paper',
            'clear_plastic': 'plastic'
        }
        
        result = {
            'material': result_map[detected_material],
            'detailed_type': detected_material,
            'confidence': round(float(confidence), 3),
            'all_scores': {
                'black_plastic': round(float(best_scores['black_plastic']), 3),
                'white_paper': round(float(best_scores['white_paper']), 3),
                'clear_plastic': round(float(best_scores['clear_plastic']), 3)
            }
        }
        
        print(f"[RESULT] {result['material']} ({detected_material})")
        print(f"[CONFIDENCE] {confidence:.3f}")
        
        # Store for mobile app
        add_to_history(result['material'], detected_material, result['confidence'])
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'references_loaded': {
            'black_plastic': len(references['black_plastic']),
            'white_paper': len(references['white_paper']),
            'clear_plastic': len(references['clear_plastic'])
        }
    })

# Test endpoint (no image needed)
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'GreenBin Waste Classification API',
        'endpoints': {
            '/classify': 'POST - Send image, get classification',
            '/health': 'GET - Check API status',
            '/history': 'GET - Get all classification history (mobile app)',
            '/latest': 'GET - Get latest classification (mobile app)',
            '/stats': 'GET - Get sorting statistics (mobile app)'
        }
    })

# ==========================================
# MOBILE APP ENDPOINTS
# ==========================================

@app.route('/history', methods=['GET'])
def get_history():
    """Get all classification history for mobile app"""
    limit = request.args.get('limit', default=50, type=int)
    return jsonify({
        'count': len(classification_history),
        'history': classification_history[:limit]
    })

@app.route('/latest', methods=['GET'])
def get_latest():
    """Get latest classification for mobile app"""
    if classification_history:
        return jsonify(classification_history[0])
    return jsonify({'message': 'No classifications yet'}), 404

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get sorting statistics for mobile app"""
    stats = {'paper': 0, 'plastic': 0, 'metal': 0, 'unknown': 0, 'total': 0}
    for entry in classification_history:
        mat = entry.get('material', 'unknown')
        if mat in stats:
            stats[mat] += 1
        stats['total'] += 1
    return jsonify(stats)

@app.route('/report', methods=['POST'])
def add_report():
    """Arduino can report sorting results here"""
    data = request.json or {}
    add_to_history(
        material=data.get('material', 'unknown'),
        detailed_type=data.get('detailed_type', 'unknown'),
        confidence=data.get('confidence', 0),
        weight=data.get('weight', 0)
    )
    return jsonify({'status': 'added'}), 201

if __name__ == '__main__':
    # Get port from environment variable (for Render.com)
    import os
    port = int(os.environ.get('PORT', 5000))
    
    print(f"Starting server on port {port}")
    print("Press CTRL+C to stop\n")
    app.run(host='0.0.0.0', port=port, debug=False)
