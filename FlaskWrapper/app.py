from flask import Flask
from urllib import request

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to Flask Wrapper"

@app.route("/search_engine/")
@app.route("/search_engine/<path:path>")
def flask1(path=""):
	return request.urlopen("http://localhost:8081/" + path).read()

@app.route("/database/")
@app.route("/database/<path:path>")
def flask2(path=""):
	return request.urlopen("http://localhost:8082/" + path).read()


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)