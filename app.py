from flask import Flask,render_template,request
import joblib
import sklearn
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

loaded_model = joblib.load ('rfc_model.sav')

app = Flask(__name__)

def get_prediction (data,loaded_model = loaded_model):
    data_model={
        'Age':data['age'],
        'Gender':data['Gender'],
        'Polyuria':data['Polyuria'],
        'Polydipsia':data['Polydipsia'],
        'sudden weight loss':data['sudden_weight_loss'],
        'weakness':data['Weakness'],
        'Polyphagia':data['Polyphagia'],
        'Genital thrush':data['Genital_Thrush'],
        'visual blurring':data['visual_blurring'],
        'Itching':data['Itching'],
        'Irritability':data['Irritability'],
        'delayed healing':data['delayed_healing'],
        'partial paresis':data['partial_paresis'],
        'muscle stiffness':data['muscle_stiffness'],
        'Alopecia':data['Alopecia'],
        'Obesity':data['Obesity'],
    }
    df = pd.DataFrame(data_model,index=[0])
    # print(df)
    prediction=loaded_model.predict(df.values)
    pred_prob = loaded_model.predict_proba(df.values)
    # print(prediction,pred_prob)
    data_model['prob']=str(int(pred_prob[0][1]*100))+" %"
    data_model['prediction']=str(prediction[0])
    data_model['email']=data['email']
    if(pred_prob[0][1]*100>=50):
        data_model['flag']=1
    else:
        data_model['flag']=0
    return data_model

@app.route("/",methods=['POST','GET'])
def hello_world():
    if request.method == 'POST':
        get_data=get_prediction(request.form)
        return render_template("result.html",risk=get_data['prob'],flag=get_data['flag'])
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)