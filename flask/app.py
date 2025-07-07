from flask import Flask, render_template, request
import pandas as pd
from preprocessing import preprocess_user_input
from model_utils import load_production_model

app = Flask(__name__)
model = load_production_model()

# Options for dropdowns
JOB_TITLES = [
    "Data Scientist", "Software Engineer", "ML Engineer", "Data Analyst"
    # Add more as needed
]
EXPERIENCE_LEVELS = ["Entry", "Mid", "Senior", "Lead"]
EMPLOYMENT_TYPES = ["FT", "PT", "CT", "IN"]
COMPANY_SIZES = ["Small", "Medium", "Large"]
COMPANY_LOCATIONS = ["US", "IN", "UK", "CA"]  # Add as needed
CURRENCIES = ["USD", "INR", "GBP", "CAD"]     # Add as needed

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        user_input = {
            "job_title": request.form['job_title'],
            "experience_level": request.form['experience_level'],
            "employment_type": request.form['employment_type'],
            "company_size": request.form['company_size'],
            "company_location": request.form['company_location'],
            "base_salary": float(request.form['base_salary']),
            "salary_currency": request.form['salary_currency'],
            "currency": request.form['currency'],
            "adjusted_total_usd": float(request.form['adjusted_total_usd']),
            "salary_in_usd": float(request.form['salary_in_usd'])
        }
        processed = preprocess_user_input(user_input)
        prediction = model.predict(processed)[0]
        return render_template('form.html', prediction=round(prediction, 2),
                               job_titles=JOB_TITLES,
                               experience_levels=EXPERIENCE_LEVELS,
                               employment_types=EMPLOYMENT_TYPES,
                               company_sizes=COMPANY_SIZES,
                               company_locations=COMPANY_LOCATIONS,
                               currencies=CURRENCIES)
    return render_template('form.html', prediction=None,
                           job_titles=JOB_TITLES,
                           experience_levels=EXPERIENCE_LEVELS,
                           employment_types=EMPLOYMENT_TYPES,
                           company_sizes=COMPANY_SIZES,
                           company_locations=COMPANY_LOCATIONS,
                           currencies=CURRENCIES)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
