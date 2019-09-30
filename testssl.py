from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
   # app.run(host='0.0.0.0',port=9005)
    app.run(host='0.0.0.0',port=9005,ssl_context=('mycert.pem', 'mykey.pem'))
