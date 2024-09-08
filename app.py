from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io

app = Flask(__name__)

# Initialize processor and model as None for lazy loading
processor = None
model = None

def load_model():
    """Lazy load the BLIP model and processor."""
    global processor, model
    if processor is None or model is None:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Load the model and processor only when needed
        load_model()

        # Get the image from the request
        image_file = request.files['image']
        img = Image.open(io.BytesIO(image_file.read()))

        # Preprocess the image for the BLIP model
        inputs = processor(images=img, return_tensors="pt")

        # Generate the text description
        output = model.generate(**inputs)
        description = processor.decode(output[0], skip_special_tokens=True)

        # Return the description
        return jsonify({'description': description})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
