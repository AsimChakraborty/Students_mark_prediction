import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import joblib
 
app=Flask(__name__)

model=joblib.load("students_mark_predictor_model.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    global df
    input_features=[int(x) for x in request.form.values()]
    features_value = np.array( input_features)
    

    if input_features[0] <0 or input_features [0] >24:
        return render_template('index.html',prediction_text='please valid hours')


    output=model.predict([features_value])[0][0].round(4)

   #input and predicted value store in csv file

    df=pd.concat([pd.DataFrame({'study hours':input_features,'Predicted output':[output]})],ignore_index=True)
    print(df)
    df.to_csv('output.csv')

    return render_template('index.html',prediction_text='you will get [{}%] marks,when you do study [{}] hours per day'.format(output,int(features_value[0])))


if __name__=="__main__":
    app.run(debug=True)