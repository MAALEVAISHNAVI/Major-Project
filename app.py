from flask import Flask, render_template, request, redirect, url_for
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load model and tokenizer on startup
model_dir = "ADHD-model/adhd_bert_model"
tokenizer_dir = "ADHD-model/adhd_bert_tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_adhd(text, tokenizer, model):
    # Debug: print input text
    print(f"Input text for prediction: {text}")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        print(f"Raw logits: {logits}")
        probs = torch.softmax(logits, dim=1)
        print(f"Softmax probabilities: {probs}")
        likelihood = probs[0][1].item()
        print(f"Predicted likelihood: {likelihood}")
        # Additional check: print predicted class
        predicted_class = torch.argmax(logits, dim=1).item()
        print(f"Predicted class index: {predicted_class}")
    return likelihood

# Home route - ADHD information
@app.route('/')
def index():
    return render_template('index.html')

# Screening route - form to input symptoms
@app.route('/screening', methods=['GET', 'POST'])
def screening():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if symptoms:
            return redirect(url_for('results', user_input=symptoms))
    return render_template('screening.html')

# Results route - display likelihood score and recommendations
@app.route('/results')
def results():
    user_input = request.args.get('user_input', '')
    likelihood_score = predict_adhd(user_input, tokenizer, model)

    if likelihood_score < 0.4:
        risk_level = 'Low'
        recommendations = 'Your responses do not indicate signs of ADHD. No further action is needed, but stay mindful of your focus and behavior over time.'
    elif 0.4 <= likelihood_score <= 0.7:
        risk_level = 'Moderate'
        recommendations = 'Some symptoms may align with ADHD. Try focus-enhancing strategies and monitor your behavior; consider seeking guidance if issues persist.'
    else:
        risk_level = 'High'
        recommendations = "Your input strongly suggests ADHD traits. It's recommended to consult a mental health professional for a detailed assessment and support."

    return render_template('results.html', user_input=user_input, score=likelihood_score, risk_level=risk_level, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
