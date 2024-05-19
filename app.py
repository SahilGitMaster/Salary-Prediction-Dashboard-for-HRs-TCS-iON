from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv('salary_data.csv')
data.drop(['capital-gain', 'capital-loss', 'education-num'], axis=1, inplace=True)

def preprocess_data(data):
    data.replace('?', np.nan, inplace=True)
    data.dropna(inplace=True)
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    X = data.drop('salary', axis=1)
    y = data['salary']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, label_encoders, scaler

X, y, label_encoders, scaler = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=2000),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=75, metric='minkowski'),
    'Decision Tree': DecisionTreeClassifier(),
    'Support Vector Classifier': SVC(),
    'Random Forest': RandomForestClassifier()
}

# Train models
for model in models.values():
    model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html', data_columns=data.columns[:-1], models=models.keys())

@app.route('/predict', methods=['POST'])
def predict():
    user_data = {column: [request.form[column]] for column in data.columns[:-1]}
    user_data_df = pd.DataFrame(user_data)
    for column in user_data_df.select_dtypes(include=['object']).columns:
        user_data_df[column] = label_encoders[column].transform(user_data_df[column])
    user_data_scaled = scaler.transform(user_data_df)
    
    selected_model = request.form['model']
    model = models[selected_model]
    prediction = model.predict(user_data_scaled)
    
    result = ' >50K' if prediction[0] == 1 else ' <=50K'
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
