# ============================================
# GREENBIN WASTE CLASSIFICATION API
# Photo comparison for ESP32-CAM
# ============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

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
            '/health': 'GET - Check API status'
        }
    })

if __name__ == '__main__':
    # Get port from environment variable (for Render.com)
    import os
    port = int(os.environ.get('PORT', 5000))
    
    print(f"Starting server on port {port}")
    print("Press CTRL+C to stop\n")
    app.run(host='0.0.0.0', port=port, debug=False)
