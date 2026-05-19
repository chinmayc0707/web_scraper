from flask import Flask, render_template

app = Flask(__name__)

# Dummy product data
PRODUCTS = [
    {
        "id": 1,
        "name": "Premium Incense Sticks (Agarbatti)",
        "description": "Hand-rolled natural incense sticks for daily prayers.",
        "price": "$5.99",
        "image": "https://placehold.co/800x600/efefef/000000?text=Incense+Sticks"
    },
    {
        "id": 2,
        "name": "Brass Brass Diya",
        "description": "Traditional brass diya for pooja rituals.",
        "price": "$12.99",
        "image": "https://placehold.co/800x600/efefef/000000?text=Brass+Diya"
    },
    {
        "id": 3,
        "name": "Copper Pooja Thali Set",
        "description": "Complete pooja thali set including roli, chawal, and diya holders.",
        "price": "$24.99",
        "image": "https://placehold.co/800x600/efefef/000000?text=Pooja+Thali"
    },
    {
        "id": 4,
        "name": "Sandalwood Paste (Chandan)",
        "description": "Pure sandalwood paste for tilak.",
        "price": "$8.49",
        "image": "https://placehold.co/800x600/efefef/000000?text=Sandalwood+Paste"
    }
]

@app.route('/')
def index():
    return render_template('index.html', featured_products=PRODUCTS[:2])

@app.route('/products')
def products():
    return render_template('products.html', products=PRODUCTS)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
