from flask import Flask, request, render_template
import numpy as np, pickle, pandas as pd

app = Flask(__name__)

model  = pickle.load(open('random_forest_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
feature_names   = pickle.load(open('feature_names.pkl', 'rb'))
label_encoders  = pickle.load(open('label_encoders.pkl', 'rb'))
categorical_cols = list(label_encoders.keys())
label_options = {col: enc.classes_.tolist() for col, enc in label_encoders.items()}

@app.route('/')
def home():
    return render_template('index.html',
                           feature_names=feature_names,
                           categorical_cols=categorical_cols,
                           label_options=label_options)

@app.route('/predict', methods=['POST'])
def predict():
    for f in feature_names:
        if request.form.get(f, '').strip() == '':
            return render_template('index.html',
                                   feature_names=feature_names,
                                   categorical_cols=categorical_cols,
                                   label_options=label_options,
                                   prediction_text="Wrong Input, All Field Must Be Filled.")
    input_vals = []
    for f in feature_names:
        val = request.form[f]
        if f in categorical_cols:
            val = label_encoders[f].transform([val])[0]
        else:
            val = float(val)
        input_vals.append(val)

    arr      = scaler.transform(np.array([input_vals]))
    predict  = model.predict(arr)[0]
    result   = "Income >50K" if predict == 1 else "Income ≤50K"

    return render_template('index.html',
                           feature_names=feature_names,
                           categorical_cols=categorical_cols,
                           label_options=label_options,
                           prediction_text=f"Prediction: {result}")

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
