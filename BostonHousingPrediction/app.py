from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from the form
        crim = float(request.form["crim"])
        zn = float(request.form["zn"])
        indus = float(request.form["indus"])
        chas = int(request.form["chas"])
        nox = float(request.form["nox"])
        rm = float(request.form["rm"])
        age = float(request.form["age"])
        dis = float(request.form["dis"])
        rad = int(request.form["rad"])
        tax = float(request.form["tax"])
        ptratio = float(request.form["ptratio"])
        b = float(request.form["b"])
        lstat = float(request.form["lstat"])

        # Make a prediction
        features = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
        prediction = model.predict(features)[0]

        return jsonify({"result": f"${prediction:.2f} thousands"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)