from flask import Flask, request, redirect, render_template
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/result', methods = ['POST'])
def result():
    age = request.form['age']
    stage = request.form['stage']
    duration = request.form['duration']
    start_cd4 = request.form['start_cd4']
    final_cd4 = request.form['final_cd4']
    return render_template('result.html', age = age, stage = stage, duration = duration, start_cd4 = start_cd4, final_cd4 = final_cd4, prediction = prediction)

df = pd.read_csv("/Users/manishmarahatta/goru-laati-smArt/FINALcsvfile.csv", index_col=0)
# print(df.columns)
# print(df.corr())

dup_df = df
# print(dup_df.head(5))

dup_df.describe(include='all')


# implementing labelEncoder
le = preprocessing.LabelEncoder()
# age_cat=le.fit_transform(df.Age)
age_cat = le.fit_transform(df.Age)
Stage_cat = le.fit_transform(df.WHO_Stage)
Duration_cat = le.fit_transform(df.Duration)
CD4start_cat = le.fit_transform(df.CD4_Start)
CD4number_cat = le.fit_transform(df.CD4_tests)
CD4last_cat = le.fit_transform(df.CD4_last)
Perform_cat = le.fit_transform(df.Performance)

# Initializing the encoded columns
# dup_df['age_cat']=age_cat
dup_df['age_cat'] = age_cat
dup_df['Stage_cat'] = Stage_cat
dup_df['Duration_cat'] = Duration_cat
dup_df['CD4start_cat'] = CD4start_cat
dup_df['CD4number_cat'] = CD4number_cat
dup_df['CD4last_cat'] = CD4last_cat
dup_df['Perform_cat'] = Perform_cat


features = dup_df.values[:, :6]
target = dup_df.values[:, 6]
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.20, random_state=20)
# print(features_train)

clf = GaussianNB()
clf.fit(features_train, target_train)
target_pred = clf.predict(features_test)


acc = accuracy_score(target_test, target_pred, normalize=True)
#print("the accuracy of our Gaussian Niave Bayes Model is: ",acc)

# e = float(input('Enter your gender: '))
# f = float(input('Enter your WHO Stage: '))
# g = float(input('Enter your Duration: '))
# h = float(input('Enter your Start CD4: '))
# i = float(input('Enter number of CD4 Done: '))
# j = float(input('Enter your recent CD4: '))
#
# list1 = [e, f, g, h, i, j]

prediction = clf.predict
print("The Performance of Patient is:")
print(prediction)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
