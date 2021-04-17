# import Flask class from the flask module
import string

from flask import Flask, request,render_template,jsonify
import joblib
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import json


# Create Flask object to run
app = Flask(__name__)

# Load the model from the file
sentiment_classification_model = joblib.load('model/sentiment_classification_model.pkl')

def validateSEntiments(listOfProductIds):
    sentimentDF= pd.read_pickle("model/Sentimantlookupfile.pkl")
    readyToRecomend=[]
    for i in listOfProductIds:
        sentimentDF_lables = sentimentDF.loc[(sentimentDF.id == i) ]
        sentimentDF_lablesPositive=sentimentDF_lables[sentimentDF_lables['class_predicted'] == 1]
        sentimentDF_lablesNegative=sentimentDF_lables[sentimentDF_lables['class_predicted'] == 0]
        PositiveCount=sentimentDF_lablesPositive['class_predicted'].count()
        NegativeCount=sentimentDF_lablesNegative['class_predicted'].count()
        if(PositiveCount < NegativeCount ):
            print("consider")
            readyToRecomend.append(i)
    return  readyToRecomend



#@app.route('/')
#def home():
    #return  render_template('home.html')

@app.route('/', methods=['GET','POST'])
def my_form_post():
    #userId = request.form['text1']
    if request.method == "POST":
        userId = request.form.get("UserId")
        #userId="0325home"
    #word = request.args.get('text1')
        recomendationDF=pd.read_pickle("model/recomendationlookupfile.pkl")
        d = recomendationDF.loc[userId].sort_values(ascending=False)[0:5]
        varifySentimentDF= pd.DataFrame(d)
        listOfProductIds=['AV13O1A8GV-KLJ3akUyj','AVpfOmKwLJeJML435GM7','AVpfL-z9ilAPnD_xWzE_']
        recommendToUser=validateSEntiments(listOfProductIds)
        print(recommendToUser)
        result =" "
        for i in recommendToUser:
            result = result +":" + i
        return "Recommended Products are :"+ result
    return render_template("home.html")

    #return jsonify(result=)


if __name__ == "__main__":
    # Start Application
    app.run()
