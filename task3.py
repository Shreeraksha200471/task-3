import pandas as pd
from sklearn.tree import DecisionTreeClassifier


data = {
    'N': [90, 85, 60, 75, 80],
    'P': [40, 45, 35, 50, 60],
    'K': [40, 35, 30, 45, 50],
    'temperature': [20, 25, 30, 22, 28],
    'humidity': [80, 70, 60, 75, 65],
    'ph': [6.5, 7.0, 6.0, 6.8, 7.2],
    'crop': ['Rice', 'Wheat', 'Maize', 'Rice', 'Wheat']
}

df = pd.DataFrame(data)


X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph']]
y = df['crop']


model = DecisionTreeClassifier()
model.fit(X, y)


new_data = pd.DataFrame([[85, 40, 40, 23, 75, 6.7]],
                        columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph'])

prediction = model.predict(new_data)

print("Recommended Crop:", prediction[0])