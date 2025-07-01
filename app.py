import os
import csv
import datetime
import torch
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image

from models.lenet import LeNet
from models.mobilenet import MobileNet
from models.resnet import ResNet
from models.vgg import VGG
from models.densenet import DenseNet 
from models.vit import VisionTransformer

# Flask 配置
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
CHECKPOINT_FOLDER = 'checkpoints'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
HISTORY_FILE = 'history.csv'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

# 模型映射
model_dict = {
    'lenet': LeNet,
    'mobilenet': MobileNet,
    'resnet': ResNet,
    'vgg': VGG,
    'densenet': DenseNet,
    'vit': VisionTransformer
}

# 图像预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 检查文件合法性
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 首页
@app.route('/')
def index():
    checkpoint_files = os.listdir(CHECKPOINT_FOLDER)
    return render_template('index.html', checkpoints=checkpoint_files)

# 识别历史记录页面
@app.route('/history')
def history():
    records = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, newline='') as f:
            reader = csv.reader(f)
            records = list(reader)
    return render_template('history.html', records=records)

# 预测接口
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'model' not in request.form:
        return jsonify({'error': 'Missing file or model name'}), 400

    file = request.files['image']
    checkpoint_name = request.form['model']
    checkpoint_path = os.path.join(CHECKPOINT_FOLDER, checkpoint_name)

    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    model_key = checkpoint_name.split('_')[0].lower()
    model_class = model_dict.get(model_key)
    if model_class is None:
        return jsonify({'error': f'Unknown model type: {model_key}'}), 400

    try:
        model = model_class(num_classes=10, input_channels=1)
        # model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        return jsonify({'error': f'Model load error: {str(e)}'}), 500

    try:
        img = Image.open(filepath).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
    except Exception as e:
        return jsonify({'error': f'Image process error: {str(e)}'}), 500

    try:
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()

        # 保存识别历史记录
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(HISTORY_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, checkpoint_name, pred, timestamp])

        return jsonify({'prediction': str(pred)})
    except Exception as e:
        print("模型加载失败：", e)
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
