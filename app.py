from flask import Flask, render_template
import sys
import os
sys.path.insert(0, '/home/nagarro/work/venvs/vgg16_code/deep-learning-models')
import run_imagenet

app = Flask(__name__)


@app.route("/")
def hello():
    return "<h1 style='color:blue'>Deep Learning COE Vgg16 Demo</h1>"

@app.route("/demo")
def demo():
    image_folder = "/home/nagarro/work/venvs/vgg16_code/flask/images/"
    image_name = [i for i in os.listdir(image_folder)]
    image_name = os.path.join(image_folder, image_name[0])
    label = run_imagenet.predict_image(image_name)
    #return render_template("index.html")
    send_data = "<p>" + label + "</p>"
    return send_data

if __name__ == "__main__":
    app.run(host='0.0.0.0')

