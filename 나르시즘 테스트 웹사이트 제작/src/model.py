import db_to_local
import pandas as pd

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

import joblib
import dill as pickle

df = db_to_local.get_db()


q_cols=['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10',
    'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20',
    'q21', 'q22', 'q23', 'q24', 'q25', 'q26', 'q27', 'q28', 'q29', 'q30',
    'q31', 'q32', 'q33', 'q34', 'q35', 'q36', 'q37', 'q38', 'q39', 'q40']
one=0
two=0
for x in q_cols:
    one+=df[x].value_counts()[1]
    two+=df[x].value_counts()[2]

df.drop([1271,8062,8268,8355,9153,9227],axis=0,inplace=True)

X=df.drop("score",axis=1)
y=df["score"]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=CatBoostRegressor(random_state=101,learning_rate= 0.2,max_depth= 8,n_estimators= 20,subsample= 0.3)
model.fit(X_train,y_train)

filename = 'model_1.pkl'

with open(filename, 'wb') as file:
    pickle.dump(model, file)

