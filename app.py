import numpy as np
from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)

class NeuralNet():
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    def relu(self, z):
        return np.maximum(0, z)
    
    def softmax(self, z):
        val = np.exp(z)/ sum(np.exp(z))
        return val

    def forward(self, data):
        z1 = self.w1.dot(data) + self.b1
        a1 = self.relu(z1)
        z2 = self.w2.dot(a1) + self.b2
        a2 = self.softmax(z2)
        return a2
    
def inNepali(val):
    words = ['क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 
             'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह', 'क्ष', 'त्र', 'ज्ञ', '०', '१', '२', '३', '४', '५', '६', '७', '८', '९']
    return words[val]

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = './images/' + imagefile.filename
    imagefile.save(image_path)

    image = Image.open(image_path).convert('L')
    image = image.resize((32, 32))
    img_array = np.array(image)
    img_array = img_array / 255.
    img_flattened = img_array.reshape(1024, 1)

    data = np.load('digit_classifier.npz')
    w1 = data['w1']
    b1 = data['b1']
    w2 = data['w2']
    b2 = data['b2']

    net = NeuralNet(w1, b1, w2, b2)
    a2 = net.forward(img_flattened)
    output = inNepali(np.argmax(a2))

    return render_template('index.html', prediction = output)

if __name__ == '__main__':
    app.run(debug=True)