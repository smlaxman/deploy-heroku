import inline as inline
import matplotlib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.metrics import f1_score


##read the data
productData=pd.read_csv('/Users/z004t01/Downloads/ACTbills/sample30.csv')
print(productData.info())
print(productData.shape)
print(productData.head(10))

print(productData["user_sentiment"].head(10))

productData['user_sentiment'].value_counts().plot.pie(figsize=(6,6),title="Distribution of reviews per sentiment",labels=['',''],autopct='%1.1f%%')
user_sentiment=["Positive","Negative"]
plt.legend(user_sentiment,loc=3)
plt.gca().set_aspect('equal')


#features = productData.drop("user_sentiment",axis=1)
#labels = productData["user_sentiment"]

productData['reviews_text'] = productData['reviews_text'].fillna('no reviews')
productData['reviews_text'] = productData['reviews_text'].astype('str')
productData['user_sentiment'] = productData['user_sentiment'].fillna('Negative')
print(productData["user_sentiment"].head(10))



#convert to lower case
productData['reviews_text'] = productData['reviews_text'].str.lower()
#Remove punctuations
#productData['reviews_text'] = productData['reviews_text'].str.replace('[^\w\s]',' ')
#Remove spaces in between words
productData['reviews_text'] = productData['reviews_text'].str.replace(' +', ' ')
#Remove Numbers
productData['reviews_text']= productData['reviews_text'].str.replace('\d+', '')
#Remove trailing spaces
productData['reviews_text'] = productData['reviews_text'].str.strip()
#Remove URLS
productData['reviews_text'] = productData['reviews_text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
#remove stop words
stop = stopwords.words('english')
#stop.extend(["racism","alllivesmatter","amp","https","co","like","people","black","white"])
productData['reviews_text'] = productData['reviews_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop ))

print("has null values")
print(productData['user_sentiment'].isnull().sum())
print("unique values are ")
print(productData["user_sentiment"].head(10))



#train test split
df_train, df_test = train_test_split(productData, test_size = 0.2, stratify =productData["user_sentiment"], random_state=21)
#get the shape of train and test split.
print(df_train.shape, df_test.shape)

#df_train['user_sentiment']= df_train['user_sentiment'].fillna(0)
#df_test['user_sentiment'] = df_test['user_sentiment'] .fillna(0)
df_train['user_sentiment'] = df_train['user_sentiment'].map({'Positive': 1, 'Negative': 0}).astype(int)
df_test['user_sentiment'] = df_test['user_sentiment'].map({'Positive': 1, 'Negative': 0}).astype(int)

tfidf_vectorizer = TfidfVectorizer(lowercase= True, max_features=15000, stop_words=ENGLISH_STOP_WORDS)
tfidf_vectorizer.fit(df_train['reviews_text'])
train_idf = tfidf_vectorizer.transform(df_train['reviews_text'])
print("train_idf")
print(type(train_idf))
print(train_idf.toarray()[0])
print(tfidf_vectorizer.get_feature_names())
test_idf = tfidf_vectorizer.transform(df_test['reviews_text'])


model_RF = RandomForestClassifier(n_estimators=100)
model_RF.fit(train_idf, df_train['user_sentiment'])
#predict the model on the train data
predict_train = model_RF.predict(train_idf)
#predict the model on the test data
predict_test = model_RF.predict(test_idf)
print("test prediction")
print(predict_test)

#f1 score on train data
print(f1_score(y_true=df_train['user_sentiment'], y_pred= predict_train))
print(f1_score(y_true=df_test['user_sentiment'], y_pred= predict_test))

# Save the Model


# Save the model as a pickle in a file
joblib.dump(model_RF, 'model/sentiment_classification_model.pkl')





# Load the model from the file
#iris_model = joblib.load('model/iris_model.pkl')


sentiment_classification_model = joblib.load('model/sentiment_classification_model.pkl')
inputData=pd.read_csv('/Users/z004t01/Downloads/ACTbills/sample30.csv')

#convert to lower case
inputData['reviews_text'] = productData['reviews_text'].str.lower()
#Remove punctuations
#productData['reviews_text'] = productData['reviews_text'].str.replace('[^\w\s]',' ')
#Remove spaces in between words
inputData['reviews_text'] = inputData['reviews_text'].str.replace(' +', ' ')
#Remove Numbers
inputData['reviews_text']= inputData['reviews_text'].str.replace('\d+', '')
#Remove trailing spaces
inputData['reviews_text'] = inputData['reviews_text'].str.strip()
#Remove URLS
inputData['reviews_text'] = inputData['reviews_text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
#remove stop words
stop = stopwords.words('english')
#stop.extend(["racism","alllivesmatter","amp","https","co","like","people","black","white"])
inputData['reviews_text'] = inputData['reviews_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop ))

tfidf_vectorizer_raw_data = TfidfVectorizer(lowercase= True,max_features=15000, stop_words=ENGLISH_STOP_WORDS)
tfidf_vectorizer_raw_data.fit(inputData['reviews_text'])
input_idf = tfidf_vectorizer_raw_data.transform(inputData['reviews_text'])

class_predicted = sentiment_classification_model.predict(input_idf)
output = "Predicted Iris Class: " + str(class_predicted)

print(type(class_predicted))

classicationbymodel= pd.DataFrame(class_predicted,columns=['class_predicted'])
print(type(classicationbymodel))



df_row = pd.concat([inputData['id'], classicationbymodel],axis=1)
#joblib.dump(df_row, 'model/Sentimantlookupfile.pkl')

pd.to_pickle(df_row, "model/Sentimantlookupfile.pkl")

print(pd.read_pickle("model/recomendationlookupfile.pkl"))

print(df_row)
