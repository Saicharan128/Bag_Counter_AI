from flask import Flask, request, jsonify
import os, tempfile, traceback
from flask_cors import CORS
from predict_gunny_bags import load_model, preprocess_image, predict_count, MODEL_PATH

app = Flask(__name__, static_url_path='', static_folder='.')
CORS(app)

# Load once at startup
try:
    model = load_model(MODEL_PATH)
    print(f"[OK] Model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"[FATAL] Could not load model at startup: {e}")
    traceback.print_exc()
    model = None

@app.route('/')
def home():
    # serves index.html from the same folder
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify(error=f"Model not loaded. Check MODEL_PATH ('{MODEL_PATH}') and file presence."), 500

        if 'image' not in request.files:
            return jsonify(error='No file uploaded under field \"image\".'), 400

        uploaded = request.files['image']
        if not uploaded or uploaded.filename.strip() == '':
            return jsonify(error='Empty filename.'), 400

        _, ext = os.path.splitext(uploaded.filename)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            uploaded.save(tmp_path)

        try:
            vec = preprocess_image(tmp_path)
            count = predict_count(model, vec)
        finally:
            try: os.unlink(tmp_path)
            except Exception: pass

        return jsonify(filename=uploaded.filename, predicted_count=int(count)), 200

    except FileNotFoundError as e:
        return jsonify(error=f"File not found: {e}"), 500
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {e}"), 500

if __name__ == '__main__':
    # Run: python app.py  -> open http://localhost:5000
    app.run(host='0.0.0.0', port=5000, debug=True)
