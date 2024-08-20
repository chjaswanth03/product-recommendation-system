from flask import Flask, render_template, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

model = load_model('model/fashion_recommendation_model.h5', custom_objects={'contrastive_loss': contrastive_loss_with_margin(margin=1)})

def get_recommendations(product_id):
    # Placeholder function to simulate fetching recommendations
    return [
        {"name": "Bag", "image": "bag.jpg"},
        {"name": "Shoes", "image": "shoes.jpg"},
        {"name": "Pencil", "image": "pencil.jpg"}
    ]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations/<int:product_id>')
def recommendations(product_id):
    recommended_products = get_recommendations(product_id)
    return jsonify(recommended_products)

if __name__ == '__main__':
    app.run(debug=True)
