import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)
api = Api(app)

df = pd.read_csv("/home/dexter/eydean/smART/FINALcsvfile.csv", index_col=0)
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

PPS = {
    'pp1': {
        'gender': 1,
        'who_stage': 1,
        'duration': 1,
        'start_cd4': 1,
        'no_cd4_done': 1,
        'recent_cd4': 1
    }
}


def abort_if_pp_doesnt_exist(pp_id):
    if pp_id not in PPS:
        abort(404, message="PatientPerformance {} doesn't exist".format(pp_id))

parser = reqparse.RequestParser(bundle_errors=True)
parser.add_argument('gender', type=float, help='Must be boolean')
parser.add_argument('who_stage', type=float)
parser.add_argument('duration', type=float)
parser.add_argument('start_cd4', type=float)
parser.add_argument('no_cd4_done', type=float)
parser.add_argument('recent_cd4', type=float)


class PatientPerformanceList(Resource):
    def get(self):
        return PPS

    def post(self):
        args = parser.parse_args()
        pp_id = int(max(PPS.keys()).lstrip('pp')) + 1
        pp_id = 'pp%i' % pp_id
        data1 = [
            args['gender'],
            args['who_stage'],
            args['duration'],
            args['start_cd4'],
            args['no_cd4_done'],
            args['recent_cd4']
        ]
        data1 = np.array(data1).reshape(1, -1)
        prediction = clf.predict(data1)
        PPS[pp_id] = {
            'gender': args['gender'],
            'who_stage': args['who_stage'],
            'duration': args['duration'],
            'start_cd4': args['start_cd4'],
            'no_cd4_done': args['no_cd4_done'],
            'recent_cd4': args['recent_cd4'],
            'result': str(prediction)
        }
        return PPS[pp_id], 201


api.add_resource(PatientPerformanceList, '/pat_per')


if __name__ == '__main__':
    app.run(debug=True)