""" 1. Import Libraries and Load Data """
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

#Load CSV data 
df1 = pd.read_csv("hangiAlanAI\\app\\student-scores.csv")
df = df1.copy()
df.head()


""" 2. Preprocessing - Data Cleaning """
#drop irrelevant columns
df.columns
df.drop(columns=['id','first_name','last_name','email'],axis=1, inplace=True)
df.head()


""" 3. Preprocessing - Encoding Categorical Features """
#Encoding Categorical Columns
gender_map = {'male': 0, 'female': 1}
part_time_job_map = {False: 0, True: 1}
career_aspiration_map = {
    'Frontend Developer': 0,
    'Android Developer': 1,
    'Data Scientist': 2,
    'Cybersecurity Specialist': 3,
    'Database Designer': 4,
    'Robotics Engineer': 5,
    'Software Tester': 6,
    'Business Systems Analyst': 7,
    'UX Designer': 8,
    'Software Architect': 9,
    'iOS Developer': 10,
    'Product Manager': 11,
    'Machine Learning Engineer': 12,
    'UI Designer': 13,
    'Database Developer': 14,
    'Game Developer': 15,
    'Backend Developer': 16,
    'Automation Engineer': 17,
    'Cybersecurity Engineer': 18,
    'Technical Writer': 19,
    'Data Engineer': 20,
    'Artificial Intelligence Engineer': 21,
    'Mobile App Tester': 22,
    'Automation Tester': 23,
    'Full Stack Developer': 24,
    'Information Security Specialist': 25,
    'IoT Engineer': 26,
    'Mobile App Developer': 27,
    'Performance Tester': 28,
    'Data Analyst': 29,
    'Network Security Specialist': 30,
    'Security Tester': 31
}
interest_areas_map = {
    'Web Development_UI/UX Design': 0,
    'Mobile App Development_Game Development': 1,
    'Data Science_Machine Learning_Artificial Intelligence': 2,
    'Cybersecurity_Cloud Computing': 3,
    'Database Management_Software Development': 4,
    'Robotics_Automation_IoT': 5,
    'Software Testing_QA': 6,
    'Business Systems Analysis_Product Management': 7,
    'User Experience Design_User Interface Design': 8,
    'Technical Writing_Software Architecture': 9
}
skills_map = {
    'HTML_CSS_JavaScript_UI Frameworks' : 0,
    'Java_Android SDK_iOS SDK_Game Engines' : 1,
    'Python_R_TensorFlow_PyTorch_Machine Learning Libraries' : 2,
    'Cybersecurity Tools_Cloud Platforms' : 3,
    'SQL_NoSQL_Databases_Software Development Tools' : 4,
    'Robotics Automation_IoT Platforms' : 5,
    'Software Testing Tools_QA Frameworks' : 6,
    'Business Systems Analysis_Product Management Tools' : 7,
    'UX/UI Design Tools_Prototyping Tools' : 8,
    'Technical Writing Tools_Software Documentation Tools' : 9
}
education_level_map = {
    'Bachelor\'s Degree': 0,
    'Master\'s Degree': 1,
    'Associate\'s Degree': 2,
    'Doctor of Law': 3,
    'Doctor of Medicine': 4
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
    'Freelance' : 0,
    'Startup' : 1,
    'Research Scientist' : 2,
    'Security Engineer' : 3,
    'Database Architect' : 4,
    'Robotics Engineer' : 5,
    'QA Automation Engineer' : 6,
    'Business Analyst' : 7,
    'UX/UI Lead' : 8,
    'Technical Writer' : 9,
    'Mobile App Developer' : 10,
    'Product Manager' : 11,
    'Data Scientist (Healthcare)' : 12,
    'UI/UX Designer' : 13,
    'Software Engineer (Backend)' : 14,
    'Game Developer' : 15,
    'Front-End Developer' : 16,
    'Automation Engineer' : 17,
    'Cloud Security Architect' : 18,
    'Software Development Engineer in Test (SDET)' : 19,
    'Full Stack Developer' : 20,
    'Data Analyst' : 21,
    'Mobile Game Developer' : 22,
    'Software Tester' : 23,
    'Web Developer' : 24,
    'Business Systems Analyst' : 25,
    'Interaction Designer' : 26,
    'Penetration Tester' : 27,
    'Software Architect' : 28,
    'AI Engineer' : 29,
    'Front-End Developer (React)' : 30,
    'QA Lead' : 31,
    'Business Systems Analyst (Project Management)' : 32
}

# Apply mapping to the DataFrame
df['gender'] = df['gender'].map(gender_map)
df['part_time_job'] = df['part_time_job'].map(part_time_job_map)
df['career_aspiration'] = df['career_aspiration'].map(career_aspiration_map)
df['interest_areas'] = df['interest_areas'].map(interest_areas_map)
df['skills'] = df['skills'].map(skills_map)
df['education_level'] = df['education_level'].map(education_level_map)
df['experience'] = df['experience'].map(experience_map)
df['career_goals'] = df['career_goals'].map(career_goals_map)

df.head()

""" Dataset Hakkında Genel Bilgiler"""
df.info()

""" "interest_areas" sütunundaki boş değerleri sil """
df_cleaned = df.dropna(subset=['interest_areas'])
df.info()


""" 4. Preprocessing - Handling Imbalanced Dataset """
#pip install imblearn
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# "career_aspiration" ve "interest_areas" sütunlarındaki boş değerleri sil
df_cleaned = df.dropna(subset=['career_aspiration', 'interest_areas'])


# Sınıf dengesizliğini kontrol etme
print(df_cleaned['career_aspiration'].unique())
print(df_cleaned['career_aspiration'].value_counts())

# (Opsiyonel) SMOTE ile aşırı örnekleme
# Label encoder oluşturma
encoder = LabelEncoder()

# "career_aspiration" sütununu kodlama
df_cleaned['career_aspiration'] = encoder.fit_transform(df_cleaned['career_aspiration'])

# Özellikleri ve hedef değişkeni ayırma
X = df_cleaned.drop('career_aspiration', axis=1)
y = df_cleaned['career_aspiration']

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


""" Models Training (Multiple Models) """
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
import pickle
# SAVE FILES
pickle.dump(scaler,open("scaler.pkl",'wb'))
pickle.dump(model,open("model.pkl",'wb'))
# Load the scaler, label encoder, and model
scaler = pickle.load(open("scaler.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))


""" Recommendation System """
#Recommendation System
import pickle
import numpy as np

# Load the scaler, label encoder, model, and class names
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
    'Database Developer', 'Machine Learning Engineer', 'Security Tester' ]

def Recommendations(gender, part_time_job, weekly_self_study_hours, career_aspiration, interest_areas, skills, education_level, experience, career_goals):

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


""" Using Example Code """
final_recommendations = Recommendations(
    gender='female',
    part_time_job=False,
    weekly_self_study_hours=20,
    career_aspiration="",
    interest_areas=0, 
    skills=0,
    education_level=0,
    experience=0,
    career_goals=3
)

print(" Top recommended career paths with probabilities:"+ "\n" +("=")*50)

for class_name, probability in final_recommendations:
    print(f"{class_name} with probability {probability}")
