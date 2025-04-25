from flask import Flask,render_template,redirect,request
import pickle as pic

app=Flask(__name__)

@app.route("/")
def index():
  return render_template("index.html",predict=0)

@app.route("/predict",methods=['POST'])
def model():
  tem=(request.form['temperature'])
  rh=float(request.form['RH'])
  ws=float(request.form['ws'])
  rain=float(request.form['rain'])
  FFMC=float(request.form['FFMC'])
  DMC=float(request.form['DMC'])
  #DC=request.form['DC']
  ISI=float(request.form['ISI'])
  #BUI=request.form['BUI']
  region=float(request.form['region'])
  classes=float(request.form['classes'])
  scaler=pic.load(open("projectScaler.pkl",'rb'))
  mod=pic.load(open("Modelridge.pkl",'rb'))
  scaleddata=scaler.transform([[tem,rh,ws,rain,FFMC,DMC,ISI,region,classes]])
  predictdata=mod.predict(scaleddata)
  print("value id ",predictdata[0],end="\n\n\n")
  return render_template("index.html",predict=predictdata[0])

if __name__=="__main__":
  app.run(host='0.0.0.0',debug=True,port=4000)