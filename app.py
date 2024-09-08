from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize processor and model as None for lazy loading
processor = None
model = None

def load_model():
    """Lazy load the BLIP model and processor."""
    global processor, model
    if processor is None or model is None:
        logging.debug("Loading BLIP model and processor...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        logging.debug("Model and processor loaded.")

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        logging.debug("Received image upload request.")
        load_model()  # Load model only when needed

        # Get the image from the request
        image_file = request.files['image']
        img = Image.open(io.BytesIO(image_file.read()))

        # Preprocess the image for the BLIP model
        logging.debug("Preprocessing image...")
        inputs = processor(images=img, return_tensors="pt")

        # Generate the text description
        logging.debug("Generating description...")
        output = model.generate(**inputs)
        description = processor.decode(output[0], skip_special_tokens=True)

        # Return the description
        logging.debug("Description generated successfully.")
        return jsonify({'description': description})

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
