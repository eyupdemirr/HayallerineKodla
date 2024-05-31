""" 1. Import Libraries and Load Data """
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

#Load CSV data 
df1 = pd.read_csv("hangiDilAI\\student-scores.csv")
df = df1.copy()
df.head()


""" 3. Preprocessing - Encoding Categorical Features """
#Encoding Categorical Columns
gender_map = {'male': 0, 'female': 1}
programming_stack = {
    'Python': 0,
    'Django': 1,
    'Flask': 2,
    'FastAPI': 3,
    'Tornado': 4,
    'Pandas': 5,
    'NumPy': 6,
    'Matplotlib': 7,
    'Seaborn': 8,
    'Scikit-learn': 9,
    'TensorFlow': 10,
    'Keras': 11,
    'PyTorch': 12,
    'NLTK': 13,
    'BeautifulSoup': 14,
    'Requests': 15,
    'SciPy': 16,
    'Statsmodels': 17,
    'XGBoost': 18,
    'LightGBM': 19,
    'CatBoost': 20,
    'H2O.ai': 21,
    'Theano': 22,
    'Scrapy': 23,
    'Lxml': 24,
    'Newspaper3k': 25,
    'Tkinter': 26,
    'PyQt': 27,
    'PyGTK': 28,
    'Kivy': 29,
    'wxPython': 30,
    'Selenium': 31,
    'PyAutoGUI': 32,
    'PyTest': 33,
    'Fabric': 34,
    'Pygame': 35,
    'Panda3D': 36,
    'Cocos2d': 37,
    "Ren'Py": 38,
    'JavaScript': 39,
    'React.js': 40,
    'Angular': 41,
    'Vue.js': 42,
    'Ember.js': 43,
    'Svelte': 44,
    'Backbone.js': 45,
    'Mithril': 46,
    'Node.js': 47,
    'Express.js': 48,
    'Meteor.js': 49,
    'Koa.js': 50,
    'Sails.js': 51,
    'Nest.js': 52,
    'D3.js': 53,
    'Chart.js': 54,
    'Highcharts': 55,
    'Plotly.js': 56,
    'Sigma.js': 57,
    'Anime.js': 58,
    'Three.js': 59,
    'GSAP': 60,
    'Mo.js': 61,
    'Java': 62,
    'Spring Boot': 63,
    'Spring MVC': 64,
    'Spring Security': 65,
    'Spring Data': 66,
    'Spring Batch': 67,
    'Spring Cloud': 68,
    'Android SDK': 69,
    'Retrofit': 70,
    'Picasso': 71,
    'Glide': 72,
    'Dagger': 73,
    'RxJava': 74,
    'Room': 75,
    'C#': 76,
    'ASP.NET': 77,
    'Entity Framework': 78,
    'WPF': 79,
    'Xamarin': 80,
    'Blazor': 81,
    'SignalR': 82,
    'ASP.NET Core': 83,
    'Entity Framework Core': 84,
    'Razor Pages': 85,
    'NancyFX': 86,
    'PHP': 87,
    'Laravel': 88,
    'Symfony': 89,
    'Zend Framework': 90,
    'Phalcon': 91,
    'CakePHP': 92,
    'CodeIgniter': 93,
    'Ruby': 94,
    'Hanami': 95,
    'Sinatra': 96,
    'Ruby on Rails': 97,
    'Swift': 98,
    'UIKit': 99,
    'SwiftUI': 100,
    'Core Data': 101,
    'Kotlin': 102,
    'Ktor': 103,
    'Koin': 104,
    'Dart': 105,
    'Flutter': 106,
    'SQL': 107,
    'PostgreSQL': 108,
    'SQLite': 109,
    'Oracle': 110,
    'Microsoft SQL Server': 111,
    'MongoDB': 112,
    'Firebase': 113,
    'Cassandra': 114,
    'DynamoDB': 115,
    'CouchDB': 116,
    'Neo4j': 117,
    'ArangoDB': 118,
    'HTML': 119,
    'CSS Frameworks': 120,
    'Bootstrap': 121,
    'Tailwind CSS': 122,
    'CSS Preprocessors': 123,
    'Sass': 124,
    'Less': 125,
    'PostCSS': 126,
    'Bulma': 127,
    'jQuery': 128,
    'Redux': 129,
    'MobX': 130,
    'Recoil': 131,
    'Webpack': 132,
    'Parcel': 133,
    'Rollup': 134,
    'Git': 135,
    'GitHub': 136,
    'GitLab': 137,
    'Bitbucket': 138,
    'SourceTree': 139,
    'TortoiseGit': 140,
    'Docker': 141,
    'Kubernetes': 142,
    'OpenShift': 143,
    'Heroku': 144,
    'Cloud Platform': 145,
    'Deployment': 146,
    'Vercel': 147,
    'Netlify': 148,
    'AWS': 149,
    'EC2': 150,
    'S3': 151,
    'Lambda': 152,
    'RDS': 153,
    'DynamoDB': 154,
    'CloudFormation': 155,
    'CloudFront': 156,
    'Sagemaker': 157,
    'Azure': 158,
    'Azure App Service': 159,
    'Azure Functions': 160,
    'Azure SQL Database': 161,
    'Azure DevOps': 162,
    'Azure Kubernetes Service (AKS)': 163,
    'Google Cloud Platform': 164,
    'Google App Engine': 165,
    'Google Compute Engine': 166,
    'Google Cloud Functions': 167,
    'Google Cloud Storage': 168,
    'BigQuery': 169,
    'Cloud Pub/Sub': 170,
    'Firestore': 171
 }

skills_map = {
    'HTML': 0,
    'CSS': 1,
    'JavaScript': 2,
    'Pandas': 3,
    'NumPy': 4,
    'Matplotlib': 5,
    'Scikit-learn': 6,
    'TensorFlow': 7,
    'BeautifulSoup': 8,
    'HTTP Requests': 9,
    'Tkinter': 10,
    'PyQt': 11,
    'Selenium': 12,
    'PyAutoGUI': 13,
    'Pygame': 14,
    'Python': 15,
    'Chart.js': 16,
    'Anime.js': 17,
    'Spring': 18,
    'AndroidSDK': 19,
    'Java': 20,
    'C#': 21,
    'PHP': 22,
    'Ruby': 23,
    'Core Data': 24,
    'Swift': 25,
    'Kotlin': 26,
    'Flutter': 27,
    'IoT': 28,
    'Dart': 29,
    'Provider': 30,
    'Riverpod': 31,
    'Bloc': 32,
    'SQL': 33,
    'Redux': 34,
    'Webpack': 35,
    'Git': 36,
    'Docker': 37,
    'Heroku': 38,
    'AWS': 39,
    'Azure': 40,
    'Google Cloud Platform': 41,
    'C': 42
}

experience_map = {
    '0 years' : 0,
    '1 years' : 1,
    '2 years' : 2,
    '3 years' : 3,
    '4 years' : 4,
    '5 years' : 5,
    '6 years' : 6,
    '7 years' : 7,
    '8 years' : 8,
    '9 years' : 9
}
career_goals_map = {
    'Web Development': 0,
    'Data Science': 1,
    'Machine Learning': 2,
    'Web Scraping': 3,
    'GUI Development': 4,
    'Automation': 5,
    'Game Development': 6,
    'Frontend Frameworks': 7,
    'Backend Frameworks': 8,
    'Data Visualization': 9,
    'Animation': 10,
    'Spring Framework': 11,
    'Android Development': 12,
    '.NET Framework': 13,
    '.NET Core': 14,
    'PHP Development': 15,
    'Ruby on Rails': 16,
    'Swift Development': 17,
    'Kotlin Development': 18,
    'Dart Development': 19,
    'Flutter Development': 20,
    'SQL': 21,
    'HTML': 22,
    'CSS': 23,
    'JavaScript': 24,
    'Redux': 25,
    'Webpack': 26,
    'Git': 27,
    'Docker': 28,
    'Heroku': 29,
    'AWS': 30,
    'Azure': 31,
    'Google Cloud Platform': 32
}


# Apply mapping to the DataFrame
df['gender'] = df['gender'].map(gender_map)
df['programming_stack'] = df['programming_stack'].map(programming_stack)
df['skills'] = df['skills'].map(skills_map)
df['experience'] = df['experience'].map(experience_map)
df['career_goals'] = df['career_goals'].map(career_goals_map)

df.head()

""" Dataset Hakkında Genel Bilgiler"""
df.info()


""" 4. Preprocessing - Handling Imbalanced Dataset """
#pip install imblearn
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# "programming_stack" boş değerleri sil
df_cleaned = df.dropna(subset=['skills','career_goals'])
df.info()

# Sınıf dengesizliğini kontrol etme
print(df_cleaned['programming_stack'].unique())
print(df_cleaned['programming_stack'].value_counts())

# (Opsiyonel) SMOTE ile aşırı örnekleme
# Label encoder oluşturma
encoder = LabelEncoder()

# "programming_stack" sütununu kodlama
df_cleaned['programming_stack'] = encoder.fit_transform(df_cleaned['programming_stack'])

# Özellikleri ve hedef değişkeni ayırma
X = df_cleaned.drop('programming_stack', axis=1)
y = df_cleaned['programming_stack']

# SMOTE uygula (isteğe bağlı)
# SMOTE nesnesi oluşturma
smote = SMOTE(random_state=42)

# SMOTE ile aşırı örnekleme
if True:
    X_resampled, y_resampled = smote.fit_resample(X, y)
else:
    X_resampled = X
    y_resampled = y


""" 5. Preprocessing - Train Test Split """
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size=0.2, random_state=42)
X_train.shape


""" 6. Preprocessing - Feature Scaling """
from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled.shape


"""Models Training (Multiple Models)"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
#pip install xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Define models
models = {
  "Logistic Regression": LogisticRegression(),
  # ... (other models omitted for brevity)
  "XGBoost Classifier": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Train and evaluate each model
for name, model in models.items():
  print("="*50)
  print("Model:", name)
  # Train the model
  model.fit(X_train_scaled, y_train)

  # Predict on test set
  y_pred = model.predict(X_test_scaled)

  # Calculate metrics
  accuracy = accuracy_score(y_test, y_pred)
  classification_rep = classification_report(y_test, y_pred)
  conf_matrix = confusion_matrix(y_test, y_pred)

  # Print metrics
  print("Accuracy:", accuracy)
  print("Classification Report:\n", classification_rep)
  print("Confusion Matrix:\n", conf_matrix)


"""Model Selection (Random Forest)"""

model = RandomForestClassifier()

model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)

# Calculate metrics
print("Accuracy: ",accuracy_score(y_test, y_pred))
print("Report: ",classification_report(y_test, y_pred))
print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))


""" Single Input Predictions """
# test 1
print("Actual Label :", y_test.iloc[10])
print("Model Prediction :",model.predict(X_test_scaled[10].reshape(1,-1))[0])
if y_test.iloc[10]==model.predict(X_test_scaled[10].reshape(1,-1)):
    print("Wow! Model doing well.....")
else:
    print("not sure......")
    
# test 2
print("Actual Label :", y_test.iloc[300])
print("Model Prediction :",model.predict(X_test_scaled[300].reshape(1,-1))[0])
if y_test.iloc[10]==model.predict(X_test_scaled[10].reshape(1,-1)):
    print("Wow! Model doing well.....")
else:
    print("not sure......")

# test 3
print("Actual Label :", y_test.iloc[23])
print("Model Prediction :",model.predict(X_test_scaled[23].reshape(1,-1))[0])
if y_test.iloc[10]==model.predict(X_test_scaled[10].reshape(1,-1)):
    print("Wow! Model doing well.....")
else:
    print("not sure......")


""" Saving & Load Files """
import os
import pickle

# Define the directory path for model files
MODEL_DIR = os.path.dirname(__file__)

# SAVE FILES
pickle.dump(scaler, open(os.path.join(MODEL_DIR, "scaler.pkl"), 'wb'))
pickle.dump(model, open(os.path.join(MODEL_DIR, "model.pkl"), 'wb'))

# Load the scaler, label encoder, and model
scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), 'rb'))
model = pickle.load(open(os.path.join(MODEL_DIR, "model.pkl"), 'rb'))


""" Recommendation System """
#Recommendation System
import os
import pickle
import numpy as np

# Define the directory path for model files
MODEL_DIR = os.path.dirname(__file__)

# Load the scaler, label encoder, model, and class names
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

def Recommendations(gender, programming_stack, skills, experience, career_goals):

    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0

    # Create feature array
    feature_array = np.array([[gender_encoded, skills, experience, career_goals]])

    # Scale features
    scaled_features = scaler.transform(feature_array)

    # Predict using the model
    probabilities = model.predict_proba(scaled_features)

    # Get top five predicted classes along with their probabilities
    top_classes_idx = np.argsort(-probabilities[0])[:5]
    top_classes_names_probs = [(class_names[idx], probabilities[0][idx]) for idx in top_classes_idx]

    return top_classes_names_probs


""" Using Example Code """
final_recommendations = Recommendations(
    gender='male',
    programming_stack="",
    skills=0,
    experience=0,
    career_goals=0
)

print(" Top recommended career paths with probabilities:"+ "\n" +("=")*50)

for class_name, probability in final_recommendations:
    print(f"{class_name} with probability {probability}")
