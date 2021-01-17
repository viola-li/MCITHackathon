from flask import Flask, request, render_template
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64
import PIL
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import time
import functools

app = Flask(__name__)

# Function to load an image from a file, and add a batch dimension.
def load_img(path_to_img):
  img = tf.io.read_file(path_to_img)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = img[tf.newaxis, :]

  return img

# Function to pre-process by resizing an central cropping it.
def preprocess_image(image, target_dim):
  # Resize the image so that the shorter dimension becomes 256px.
  shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
  short_dim = min(shape)
  scale = target_dim / short_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  image = tf.image.resize(image, new_shape)

  # Central crop the image.
  image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

  return image

# Function to run style prediction on preprocessed style image.
def run_style_predict(preprocessed_style_image):
  # Load the model.
  interpreter = tf.lite.Interpreter(model_path=style_predict_path)

  # Set model input.
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

  # Calculate style bottleneck.
  interpreter.invoke()
  style_bottleneck = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
      )()

  return style_bottleneck

# Run style transform on preprocessed style image
def run_style_transform(style_bottleneck, preprocessed_content_image):
  # Load the model.
  interpreter = tf.lite.Interpreter(model_path=style_transform_path)

  # Set model input.
  input_details = interpreter.get_input_details()
  interpreter.allocate_tensors()

  # Set model inputs.
  interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
  interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
  interpreter.invoke()

  # Transform content image.
  stylized_image = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
      )()

  return stylized_image

def select_degree(degree_txt = 'full'):
    if degree_txt == 'full':
        degree = 0
    elif degree_txt == 'half':
        degree = 0.5
    else:
        degree = 0.9
    return degree

def style_mixing(content_path,style_path,degree=0):
    #perform final style mixing
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    preprocessed_content_image = preprocess_image(content_image, 384)
    preprocessed_style_image = preprocess_image(style_image, 256)

    style_bottleneck = run_style_predict(preprocessed_style_image)
    style_bottleneck_content = run_style_predict(preprocess_image(content_image, 256))

    #get degree
    content_blending_ratio = degree

    style_bottleneck_blended=(content_blending_ratio * style_bottleneck_content \
                            + (1 - content_blending_ratio) * style_bottleneck)
    stylized_image_blended=(run_style_transform(style_bottleneck_blended,
                                                preprocessed_content_image))
    return stylized_image_blended

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

@app.route('/success', methods = ['POST', 'GET'])  
def success():  
    if request.method == 'POST':  
        content_img = request.files['content_img']
        style_img = request.files['style_img']
        upload_dir = './upload/'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        content_path = os.path.join(upload_dir,content_img.filename)#set the path to the /upload/ folder
        content_img.save(content_path)  #save the content image file

        style_path = os.path.join(upload_dir,style_img.filename)#set the path to the /upload/ folder
        style_img.save(style_path)  #save the content image file

        degree_txt = request.files['degree'].read().decode("utf-8") 
        degree = select_degree(degree_txt)

        mixed_img = style_mixing(content_path,style_path,degree=degree)
        mixed_img = tf.cast(tf.squeeze(mixed_img,axis=0)*255,tf.int8)
        mixed_img = Image.fromarray(mixed_img.numpy(),mode='RGB')
        mixed_img_path = '../search_engine/static/mixed_img.png'
        mixed_img.save(mixed_img_path)
        #data_url = base64.b64encode(open(mixed_img_path,'rb').read()).decode('utf-8') #read the uploaded file
        #data_url = base64.b64encode(mixed_img).decode('utf-8')
        #img_tag = '<img src="data:image/jpg;base64,{0}" style="margin-left: 35%">'.format(data_url) #create image tag
        return render_template("render.html")
        #return img_tag #display


if __name__=="__main__":
    style_predict_path = tf.keras.utils.get_file('style_predict_original.tflite', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite')
    style_transform_path = tf.keras.utils.get_file('style_transform_original.tflite', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite')
    
    app.run(host='0.0.0.0', port=8082, debug=True)  