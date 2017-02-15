from flask import Flask, render_template ,request
import sys
from werkzeug import secure_filename
import os
import run_imagenet
import uuid

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('index.html')

# @app.route("/demo")
# def demo():
#     print("Full Path of Images Folder " + os.path.join(os.getcwd(),"images"))
#     #image_folder="/home/nagarro/work/venvs/vgg16_code/flask/images/"
#     image_name = [i for i in os.listdir(os.path.join(os.getcwd(),"images"))]
#
#     image_name = os.path.join(os.getcwd(),'images', image_name[3])
#     print("Image Name : " + image_name)
#     label = run_imagenet.predict_image(image_name)
#     #return render_template("index.html")
#     send_data = "<p>" + label + "</p>"
#     return send_data

@app.route('/demo', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      print('uploaded images : ', f.filename)
      image_name = str(uuid.uuid4())
      print("Image Name : " + image_name)
      image_folder = 'images'
      image_full_path = os.path.join(os.getcwd(), image_folder , image_name )
      f.save(image_full_path)

      label = run_imagenet.predict_image(image_name)
      os.remove(image_full_path)
      return label

if __name__ == "__main__":
    app.run(host='0.0.0.0')

