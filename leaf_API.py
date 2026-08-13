from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (from Flutter app)

# Load model
model = tf.keras.models.load_model("final_leaf_disease_model.keras")

# Class names
class_names = ['Apple_Applescab', 'AppleBlack_rot', 'Apple_Cedar_applerust', 'Applehealthy', 'Blueberryhealthy',
               'Cherry(includingsour)Powderymildew', 'Cherry(including_sour)healthy',
               'Corn(maize)_Cercospora_leaf_spot Gray_leafspot', 'Corn(maize)_Commonrust',
               'Corn(maize)Northern_LeafBlight', 'Corn(maize)healthy', 'GrapeBlack_rot', 'GrapeEsca(BlackMeasles)',
               'GrapeLeafblight(Isariopsis_Leaf_Spot)', 'Grapehealthy', 'OrangeHaunglongbing_(Citrus_greening)',
               'Peach_Bacterialspot', 'Peachhealthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,bellhealthy',
               'Potato_Earlyblight', 'PotatoLate_blight', 'Potatohealthy', 'Raspberryhealthy', 'Soybeanhealthy',
               'SquashPowdery_mildew', 'Strawberry_Leafscorch', 'Strawberryhealthy', 'Tomato_Bacterialspot',
               'TomatoEarly_blight', 'Tomato_Lateblight', 'TomatoLeaf_Mold', 'Tomato_Septoria_leafspot',
               'TomatoSpider_mites Two-spotted_spider_mite', 'Tomato_TargetSpot',
               'TomatoTomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaicvirus', 'Tomatohealthy']

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    try:
        # Load and preprocess image
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)
        predicted_class = class_names[np.argmax(preds)]
        confidence = float(np.max(preds))

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ FIXED ENTRY POINT
if __name__ == "__main__":
    app.run(debug=True)
