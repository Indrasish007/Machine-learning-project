from flask import Flask as fl, render_template
app = fl(__name__)

@app.route('/')
def hello_world():
    # return "hello world"
    return render_template("index.html")

@app.route('/products')
def products():
    return "This is product page"
if __name__ == "__main__":
    app.run(debug=True)


