import numpy as np
from flask import Flask, render_template,request
import pickle
import joblib
import warnings

app = Flask(__name__)


with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=UserWarning)
      model = joblib.load('model_fish2.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])

def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]

    return render_template('index.html', prediction_text='The fish price is  :{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
