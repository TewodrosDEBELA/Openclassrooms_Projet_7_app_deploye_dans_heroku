# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 23:41:36 2022

@author: Tewod
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from joblib import load
import json
from lightgbm import LGBMClassifier


app = Flask(__name__)

#API_url = "http://127.0.0.1:5000/credit/" + id_client

# tell Flask to use the above defined config

clf = pickle.load(open('lgbm_classifier.pickle', 'rb'))

sample = pd.read_csv(r'C:\Users\Tewod\OneDrive\Bureau\Openclassrooms\projets\projet7\OpenClassrooms-Project7\X_test_sample.csv', index_col='SK_ID_CURR', encoding ='utf-8')

#X=sample.copy()

clf.predict_proba(sample.loc[[362145]])[:,1]


@app.route('/home')
def home():
    return 'Hello World'
    

@app.route('/credit/', methods=['GET'])

@app.route('/credit/<int:id_client>' , methods=['GET'])


def credit(id_client):
    
        id = id_client
        score = clf.predict_proba(sample.loc[[id]])[:,1]
        predict = clf.predict(sample.loc[[id]])

        # round the predict proba value and set to new variable
        percent_score = score*100
        
        id_risk = np.round(percent_score, 3)
        # create JSON object
        output = {'prediction': int(predict), 'client risk in %': float(id_risk)}

        print('Nouvelle Prédiction : \n', output)
        
        return output
    
  #API_url = "http://127.0.0.1:5000/credit/" + '<int:id_client>'     
    
if __name__ == '__main__':
     app.run()