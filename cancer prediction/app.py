import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('brest_cancer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['texture_mean', 'smoothness_mean', 'compactness_mean', 'concave_points_mean',
    'symmetry_mean', 'fractal_dimension_mean', 'texture_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'texture_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

    df = pd.DataFrame(features_value, columns= features_name)
    output = model.predict(df)

    if output == 0:
        res_val = "no breast cancer"
    else:
        res_val = "breast cancer"

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))
if __name__ == '__main__':
    app.run()