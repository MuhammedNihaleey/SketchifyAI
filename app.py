import os
import torch
from flask import Flask, render_template, request, jsonify
import base64
import traceback
from torchvision import transforms
from model import Generator
from io import BytesIO
from PIL import ImageEnhance
import cv2
import numpy as np
from PIL import Image, ImageEnhance


# Initialize Flask application
app = Flask(__name__)

# Define hyperparameters
input_dim = 100  # Size of noise vector
num_classes = 6  # Number of object categories
embedding_dim = 50  # Dimensionality of class embeddings

# Initialize the generator model
generator = Generator(input_dim, 1, num_classes, embedding_dim * 2)

# Load the trained model weights
generator.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))
generator.eval()

# Define function to map text descriptions to class labels
def map_text_to_class(text_description):
    class_mapping = {
        'alarmclock': 0,
        'apple': 1,
        'bicycle': 2,
        'calculator': 3,
        'cigarette': 4,
        'cow': 5
    }
    return class_mapping.get(text_description.lower(), -1)

# Route for index page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_sketch', methods=['POST'])
def generate_sketch():
    try:
        # Get JSON data from the request
        text_description = request.json.get('text_description', '')
        class_label = map_text_to_class(text_description)

        # Generate random noise
        noise = torch.randn(1, input_dim)

        # Generate sketch using the model
        with torch.no_grad():
            generated_image = generator(noise, torch.tensor([class_label], dtype=torch.long))

        # Convert the generated image tensor to a PIL image
        generated_image = generated_image.squeeze(0).detach().cpu()
        generated_image = (generated_image + 1) / 2  # Scale image to range [0, 1]
        pil_image = transforms.ToPILImage()(generated_image)

        # Convert PIL image to grayscale NumPy array
        image_np = np.array(pil_image.convert('L'))

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image_np = np.array(pil_image.convert('L'))
        clahe_image_np = clahe.apply(image_np)
        clahe_image = Image.fromarray(clahe_image_np)

        # Convert back to PIL image
        clahe_image = Image.fromarray(clahe_image_np)

        # Apply a milder contrast enhancement
        enhancer = ImageEnhance.Contrast(clahe_image)
        enhanced_image = enhancer.enhance(6.0)  # Adjust contrast factor as needed

        # Save the enhanced image to a temporary buffer
        buffer = BytesIO()
        enhanced_image.save(buffer, format='PNG')
        buffer.seek(0)

        # Encode the image in base64
        sketch_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        # Prepare the response with the base64 encoded image
        response = {'sketch': sketch_base64}

        # Return the response as JSON
        return jsonify(response)

    except Exception as e:
        print(traceback.format_exc())  # Print the traceback to debug errors
        return jsonify({'error': 'An error occurred during sketch generation'}), 500


if __name__ == '__main__':
    app.run(debug=True)
