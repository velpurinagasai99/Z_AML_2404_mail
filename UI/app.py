from flask import Flask
from datetime import datetime
from flask import render_template


app = Flask(__name__)

@app.route("/hello/")
@app.route("/hello/<name>")
def hello_there(name = None):
    return render_template(
        "hello_there.html",
        name=name,
        date=datetime.now()
    )