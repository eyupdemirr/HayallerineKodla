from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

MODEL_DIR = os.path.dirname(__file__)

scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), 'rb'))
model = pickle.load(open(os.path.join(MODEL_DIR, "model.pkl"), 'rb'))
class_names = [
    'HTML', 'CSS', 'JavaScript', 'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Scikit-learn',
    'TensorFlow', 'Keras', 'PyTorch', 'NLTK', 'BeautifulSoup', 'Requests', 'SciPy', 'Statsmodels',
    'XGBoost', 'LightGBM', 'CatBoost', 'H2O.ai', 'Theano', 'Scrapy', 'Lxml', 'Newspaper3k',
    'Tkinter', 'PyQt', 'PyGTK', 'Kivy', 'wxPython', 'Selenium', 'PyAutoGUI', 'PyTest', 'Fabric',
    'Pygame', 'Panda3D', 'Cocos2d', "Ren'Py", 'React.js', 'Angular', 'Vue.js', 'Ember.js', 'Svelte',
    'Backbone.js', 'Mithril', 'Node.js', 'Express.js', 'Meteor.js', 'Koa.js', 'Sails.js', 'Nest.js',
    'D3.js', 'Chart.js', 'Highcharts', 'Plotly.js', 'Sigma.js', 'Anime.js', 'Three.js', 'GSAP', 'Mo.js',
    'Spring Boot', 'Spring MVC', 'Spring Security', 'Spring Data', 'Spring Batch', 'Spring Cloud', 'Android SDK',
    'Retrofit', 'Picasso', 'Glide', 'Dagger', 'RxJava', 'Room', 'ASP.NET', 'Entity Framework', 'WPF', 'Xamarin',
    'Blazor', 'SignalR', 'ASP.NET Core', 'Entity Framework Core', 'Razor Pages', 'NancyFX', 'Laravel', 'Symfony',
    'Zend Framework', 'Phalcon', 'CakePHP', 'CodeIgniter', 'Hanami', 'Sinatra', 'Ruby on Rails', 'UIKit', 'SwiftUI',
    'Core Data', 'Kotlin', 'Ktor', 'Koin', 'Dart', 'Flutter', 'SQL', 'PostgreSQL', 'SQLite', 'Oracle', 'Microsoft SQL Server',
    'MongoDB', 'Firebase', 'Cassandra', 'DynamoDB', 'CouchDB', 'Neo4j', 'ArangoDB', 'CSS Frameworks', 'Bootstrap',
    'Tailwind CSS', 'CSS Preprocessors', 'Sass', 'Less', 'PostCSS', 'Bulma', 'jQuery', 'Redux', 'MobX', 'Recoil',
    'Webpack', 'Parcel', 'Rollup', 'Git', 'GitHub', 'GitLab', 'Bitbucket', 'SourceTree', 'TortoiseGit', 'Docker',
    'Kubernetes', 'OpenShift', 'Heroku', 'Cloud Platform', 'Deployment', 'Vercel', 'Netlify', 'EC2', 'S3', 'Lambda',
    'RDS', 'CloudFormation', 'CloudFront', 'Sagemaker', 'Azure App Service', 'Azure Functions', 'Azure SQL Database',
    'Azure DevOps', 'Azure Kubernetes Service (AKS)', 'Google App Engine', 'Google Compute Engine', 'Google Cloud Functions',
    'Google Cloud Storage', 'BigQuery', 'Cloud Pub/Sub', 'Firestore'
]

# main Function
def Recommendations(gender,  skills, experience, career_goals):
    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0

    feature_array = np.array([[gender_encoded, skills, experience, career_goals]])
    scaled_features = scaler.transform(feature_array)
    probabilities = model.predict_proba(scaled_features)

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
        skills = int(request.form['skills'])
        experience = int(request.form['experience'])
        career_goals = int(request.form['career_goals'])

        recommendations = Recommendations(gender, skills, experience, career_goals)

        return render_template('results.html', recommendations=recommendations)
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
