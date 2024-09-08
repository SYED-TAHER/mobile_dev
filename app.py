from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)

# Load the pre-trained BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Check if an image is provided in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        # Get the image from the request
        image_file = request.files['image']
        print("Received image file:", image_file.filename)  # Debugging log
        
        # Open the image file
        img = Image.open(io.BytesIO(image_file.read()))

        # Preprocess the image for the BLIP model
        inputs = processor(images=img, return_tensors="pt")

        # Generate the text description
        output = model.generate(**inputs)
        description = processor.decode(output[0], skip_special_tokens=True)

        print("Generated description:", description)  # Debugging log
        
        # Return the description as JSON
        return jsonify({'description': description})

    except Exception as e:
        # Log and return any errors encountered
        print("Error in processing:", str(e))  # Debugging log
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start the Flask app on all available IP addresses on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
