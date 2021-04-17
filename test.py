import pandas as pd
import json



def my_form_post():
    #userId = request.form['text1']
    userId="0325home"
    #word = request.args.get('text1')
    recomendationDF=pd.read_pickle("model/recomendationlookupfile.pkl")
    d = recomendationDF.loc[userId].sort_values(ascending=False)[0:5]
    print(d)
    listOfProductIds= (d.index.values.tolist())
    #istOfProductIds=['AV13O1A8GV-KLJ3akUyj','AVpfOmKwLJeJML435GM7','AVpfL-z9ilAPnD_xWzE_']
    recommendToUser=validateSEntiments(listOfProductIds)
    print(recommendToUser)
    count=0
    for i in recommendToUser:
        count = count +1
        row_json= json.dumps({"recommand: "+str(count): i })
    print(row_json)



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





if __name__ == "__main__":
# Start Application
#app.run()
    my_form_post()