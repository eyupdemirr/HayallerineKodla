from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the scaler, model, and class names
scaler = pickle.load(open("scaler.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))
class_names = [
    'Frontend Developer', 'Android Developer', 'Data Scientist', 'Cybersecurity Specialist',
    'Database Designer', 'Robotics Engineer', 'Software Tester', 'Business Systems Analyst',
    'UX Designer', 'Software Architect', 'iOS Developer', 'Product Manager',
    'Machine Learning Engineer', 'UI Designer', 'Database Developer', 'Game Developer',
    'Backend Developer', 'Automation Engineer', 'Cybersecurity Engineer', 'Technical Writer',
    'Data Engineer', 'Artificial Intelligence Engineer', 'Mobile App Tester', 'Automation Tester',
    'Full Stack Developer', 'Business Systems Analyst', 'UI Designer', 'Information Security Specialist',
    'Software Architect', 'IoT Engineer', 'Mobile App Developer', 'Performance Tester', 'Product Manager',
    'Full Stack Developer', 'Android Developer', 'Data Analyst', 'Network Security Specialist',
    'Database Developer', 'Machine Learning Engineer', 'Security Tester'
]

# Recommendation System
def Recommendations(gender, part_time_job, weekly_self_study_hours, interest_areas, skills, education_level, experience, career_goals):
    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0

    # Create feature array
    feature_array = np.array([[gender_encoded, part_time_job_encoded, weekly_self_study_hours, interest_areas, skills, education_level, experience, career_goals]])

    # Scale features
    scaled_features = scaler.transform(feature_array)

    # Predict using the model
    probabilities = model.predict_proba(scaled_features)

    # Get top five predicted classes along with their probabilities
    top_classes_idx = np.argsort(-probabilities[0])[:5]
    top_classes_names_probs = [(class_names[idx], probabilities[0][idx]) for idx in top_classes_idx]

    return top_classes_names_probs

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend')
def recommend():
    return render_template('recommend.html')

@app.route('/pred', methods=['POST'])
def pred():
    if request.method == 'POST':
        gender = request.form['gender']
        part_time_job = request.form['part_time_job'] == 'true'
        weekly_self_study_hours = int(request.form['weekly_self_study_hours'])
        interest_areas = int(request.form['interest_areas'])
        skills = int(request.form['skills'])
        education_level = int(request.form['education_level'])
        experience = int(request.form['experience'])
        career_goals = int(request.form['career_goals'])

        recommendations = Recommendations(gender, part_time_job, weekly_self_study_hours,
                                          interest_areas, skills,
                                          education_level, experience, career_goals)

        return render_template('results.html', recommendations=recommendations)
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
