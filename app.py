from flask import Flask, request, jsonify, session, url_for, redirect, render_template
from flower_form import FlowerForm
from model import make_prediction
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import os
import json
import ast


def encode(sample_json):
    COUNTRY_CODE = sample_json['COUNTRY_CODE']
    MCC_CODE = sample_json['MCC_CODE']
    US_TRAN_AMT = sample_json['US_TRAN_AMT']
    flower = [[COUNTRY_CODE, MCC_CODE, US_TRAN_AMT]]
    return str(flower)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

@app.route("/", methods=['GET','POST'])
def index():
    form = FlowerForm()

    if form.validate_on_submit():
        session['COUNTRY_CODE'] = form.COUNTRY_CODE.data
        session['MCC_CODE'] = form.MCC_CODE.data
        session['US_TRAN_AMT'] = form.US_TRAN_AMT.data

        return redirect(url_for("prediction"))
    return render_template("home.html", form=form)

@app.route('/prediction')
def prediction():
    sample_json = {'COUNTRY_CODE': float(session['COUNTRY_CODE']), 'MCC_CODE': float(session['MCC_CODE']),
               'US_TRAN_AMT': float(session['US_TRAN_AMT'])}

    results = make_prediction(encode(sample_json))

    return render_template('prediction.html', results=results)
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8833)