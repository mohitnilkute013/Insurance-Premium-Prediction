from flask import Flask,request,render_template,jsonify
from pipelines.prediction_pipeline import CustomData,PredictPipeline
from pipelines import training_pipeline
import os
import setuptools
import distutils


application=Flask(__name__)

app=application


@app.route('/', methods=['GET', 'POST'])
def home_page():

    title = 'Insurance Premium Prediction App'

    parameters = {
            'age': 'text',
            'sex': ['male', 'female'],
            'bmi': 'text',
            'children': 'text',
            'smoker': ['yes', 'no'],
            'region': ['southeast', 'southwest', 'northwest', 'northeast']
            # Add more parameters as needed
        }


    if request.method=='GET':
        return render_template('index.html', title=title, parameters=parameters)
    else:
        data = CustomData(
            age = int(request.form.get('age')),
            sex = request.form.get('sex'),
            bmi = float(request.form.get('bmi')),
            children = int(request.form.get('children')),
            smoker = request.form.get('smoker'),
            region = request.form.get('region')
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results = f"The predicted result is: {round(pred[0], 2)}"

        return render_template('index.html', title=title, parameters=parameters, final_result=results)


@app.route('/train', methods=['POST'])
def train_model():

    completed = training_pipeline.Train()
    
    response_data = {'status': 'completed'} if completed else {'status': 'failed'}
    
    return jsonify(response_data)



if __name__=="__main__":
    app.run(host='0.0.0.0',debug=False, port= int(os.environ.get('PORT', 8080)))

