from sklearn.tree import DecisionTreeClassifier
import numpy as np
import joblib

# Dummy training data: [height, weight, chest, waist, hips]
X = np.array([
    [170, 65, 90, 75, 95],
    [160, 50, 85, 65, 90],
    [180, 80, 100, 85, 105],
    [175, 70, 95, 80, 100]
])

# Dummy labels: clothing sizes
y = ['M', 'S', 'L', 'M']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')
print("Model trained and saved as model.pkl")
