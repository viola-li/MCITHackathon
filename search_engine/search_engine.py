import sys
import requests
from flask import Flask, render_template, request, flash,url_for,redirect
import base64
import os


app = Flask(__name__)
@app.route("/")
def start():
    return render_template('index.html')


@app.route('/success', methods = ['POST', 'GET'])  
def upload_img():
    if request.method=='POST':  # User clicked submit button        
      content_img = request.files['content_img']
      content_img.name = content_img.filename
      style_img = request.files['style_img']
      style_img.name = style_img.filename
      degree = request.form.get('degree')      
      resp = requests.post(url=db_url+'/success',files={"content_img":content_img,"style_img":style_img,"degree":degree})
      return resp.content
    else:
      return  render_template('index.html') 

if __name__=="__main__":
    # determine what the URL for the database should be, port is always 8082 for DB
    if(len(sys.argv) == 2):
        db_url = 'http://' + sys.argv[1] + ':8082'
    else:
        db_url = "http://0.0.0.0:8082"  
   
    app.run(host='0.0.0.0', port=8081, debug=True)
