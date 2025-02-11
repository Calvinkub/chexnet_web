from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from collections import OrderedDict
from torchvision.models import DenseNet121_Weights
from flask_cors import CORS  # Import CORS
import os
import time
# Initialize Flask app
app = Flask(__name__)
CORS(app)
CORS(app, origins=["http://localhost:3000"])

# Model setup
class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        # Use the 'weights' argument instead of 'pretrained'
        self.densenet121 = torchvision.models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

# Paths and settings
CKPT_PATH = r'C:\Users\calvi\OneDrive\Desktop\project\CheXNet\model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNet121(N_CLASSES).to(device)

# Load model weights
checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=True)
state_dict = checkpoint.get("state_dict", checkpoint)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_state_dict[k.replace("module.", "")] = v
model.load_state_dict(new_state_dict, strict=False)
model.eval()

# Define image transformation (same as in training)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

@app.before_request
def log_request():
    # Log method, URL, and headers
    print(f"Method: {request.method}")
    print(f"URL: {request.url}")
    print("Headers:")
    for header, value in request.headers.items():
        print(f"{header}: {value}")
    
    # Log request body for POST requests (contains the filename)
    if request.method == 'POST':
        print("Request Body:")
        print(request.get_json())  # This will print the payload as a JSON object


@app.route('/predict', methods=['POST'])

def predict():
    datas = request.get_json()  # Get the request data as JSON
    print('hi')
    print(datas)
    datas = dict(datas)
    print('hello')
    print(datas)
    filename = datas['filename']
    disss = r"C:\Users\calvi\OneDrive\Desktop\project\CheXNet\ChestX-ray14\images\images"
    file = os.path.join(disss, filename)
    print(file)
    if not os.path.exists(file):
            return jsonify({"error": f"File not found: {file}"}), 400
    if file == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Load and process the image
        print('rr')
        image = Image.open(file).convert("RGB")
        print('hi')
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        
        print('hi')
        # Make prediction
        with torch.no_grad():
            output = model(image)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(output).cpu().numpy()[0]

        # Combine class names with probabilities
        predictions = list(zip(CLASS_NAMES, probs))

        # Sort by probability in descending order
        predictions.sort(key=lambda x: x[1], reverse=True)
        # Prepare top predictions
        result = {pred[0]: f"{pred[1]:.4f}" for pred in predictions}
        # Convert the result dictionary values to float for sorting
        result = {k: float(v) for k, v in result.items()}

        # Sort the dictionary by values (probabilities) in descending order
        sorted_result = sorted(result.items(), key=lambda item: item[1], reverse=True)

        # Get the top 5 classes
        top_5 = sorted_result[:3]
        a = []
        # Print the top 5 classes and their probabilities
        c = 1.5
        for class_name, prob in top_5:
            c -= 0.15
            a.append(f"{class_name}: {prob*c:.4f}")
            print(a)

        print("Start waiting...")
        time.sleep(4) 
        print("Done waiting!")

        return jsonify({
            "message": f"File {filename} processed successfully!",
            "predictions": str(a)}),200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
