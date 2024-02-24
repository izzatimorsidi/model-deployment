from flask import Flask, render_template, request
import pickle

tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])

def predict():
    email_text = request.form.get("email-content")
    tokenize_email = tokenizer.transform([email_text])  # Wrap email_text in a list
    predictions = model.predict(tokenize_email)
    predictions = 1 if predictions == 1 else -1
    return render_template("index.html", predictions=predictions, email_text=email_text)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)