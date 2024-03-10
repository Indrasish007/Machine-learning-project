from flask import Flask as fl
app = fl(__name__)

from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return "hello world"


@app.route('/products')
def products():
    return "This is product page"
if __name__ == "__main__":
    app.run(debug=True)


