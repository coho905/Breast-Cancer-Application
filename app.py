from flask import Flask, request, render_template
import pandas as pd
import joblib

# Declare a Flask app
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    # If a form is submitted
    if request.method == "POST":

        # Unpickle classifier
        clf = joblib.load("model.pkl")

        # Get values through input bars
        height = request.form.get("weight1")
        height2 = request.form.get("weight2")
        height3 = request.form.get("weight3")
        height4 = request.form.get("weight4")
        height5 = request.form.get("weight5")
        height6 = request.form.get("weight6")

        # Put inputs to dataframe
        X = pd.DataFrame([[height, height2, height3, height4, height5, height6]], columns=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean'])
        # Get prediction
        prediction = clf.predict(X)[0]
        if prediction == 0:
            prediction = 'benign'
        else:
            prediction = 'malignant'

    else:
        prediction = ""

    return render_template("website.html", output=prediction)

