from flask import Flask # Imports the Flask module and create a Flask web server from the Flask module
from q2_nmf import get_nmf  # Used for q3

app = Flask(__name__)   # __name__ refers to the current file

@app.route("/")
def home():
    return "STAT"

@app.route("/number/<some_int>", methods = ['PUT'])
def number(some_int):
    return str(int(some_int) + 100)

@app.route("/NMF", methods = ['GET'])
def nmf():
    return get_nmf()
    
if __name__ == "__main__":
    app.run(debug=True)