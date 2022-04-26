import flask
from flask import Flask, jsonify, request, render_template
import joblib
import numpy as np
import dill as pickle


def create_app():
   
    app = Flask(__name__)

    filename = 'model_1.pkl'


    # 메인 페이지 라우팅
    @app.route('/')
    def index():
        return render_template('index.html')

    # 데이터 예측
    @app.route('/predict', methods=['POST'])
    def prediction():
            with open(filename, 'rb') as f:
                model = pickle.load(f)

            q1 = request.form['q1']
            q2 = request.form['q2']
            q3 = request.form['q3']
            q4 = request.form['q4']
            q5 = request.form['q5']
            q6 = request.form['q6']
            q7 = request.form['q7']
            q8 = request.form['q8']
            q9 = request.form['q9']
            q10 = request.form['q10']
            q11 = request.form['q11']
            q12 = request.form['q12']
            q13 = request.form['q13']
            q14 = request.form['q14']
            q15 = request.form['q15']
            q16 = request.form['q16']
            q17 = request.form['q17']
            q18 = request.form['q18']
            q19 = request.form['q19']
            q20 = request.form['q20']
            q21 = request.form['q21']
            q22 = request.form['q22']
            q23 = request.form['q23']
            q24 = request.form['q24']
            q25 = request.form['q25']
            q26 = request.form['q26']
            q27 = request.form['q27']
            q28 = request.form['q28']
            q29 = request.form['q29']
            q30 = request.form['q30']
            q31 = request.form['q31']
            q32 = request.form['q32']
            q33 = request.form['q33']
            q34 = request.form['q34']
            q35 = request.form['q35']
            q36 = request.form['q36']
            q37 = request.form['q37']
            q38 = request.form['q38']
            q39 = request.form['q39']
            q40 = request.form['q40']
            gender = request.form['gender']
            age = request.form['age']


            pred = model.predict(
                [float(q1),float(q2),float(q3),float(q4),float(q5),float(q6),float(q7),float(q8),
                float(q9),float(q10),float(q11),float(q12),float(q13),float(q14),float(q15),float(q16),
                float(q17),float(q18),float(q19),float(q20),float(q21),float(q22),float(q23),float(q24),
                float(q25),float(q26),float(q27),float(q28),float(q29),float(q30),float(q31),float(q32),
                float(q33),float(q34),float(q35),float(q36),float(q37),float(q38),float(q39),float(q40),
                float(gender), float(age)]
            )
            result = np.round(pred,1)
            return render_template('result.html', result=result)

    return app
    
if __name__ == "__main__":
    app = create_app()
    app.run()