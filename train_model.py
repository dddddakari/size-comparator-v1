import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

# Enhanced training data with multiple brands and realistic measurements
# Format: [height_cm, weight_kg, chest_cm, waist_cm, hips_cm, brand]
training_data = [
    # Zara (European sizing - tends to run smaller)
    [160, 50, 82, 64, 88, 'Zara', 'XS'],
    [165, 55, 86, 68, 92, 'Zara', 'S'],
    [170, 60, 90, 72, 96, 'Zara', 'M'],
    [175, 65, 94, 76, 100, 'Zara', 'L'],
    [180, 70, 98, 80, 104, 'Zara', 'XL'],
    [162, 52, 84, 66, 90, 'Zara', 'S'],
    [168, 58, 88, 70, 94, 'Zara', 'M'],
    [172, 62, 92, 74, 98, 'Zara', 'M'],
    [178, 68, 96, 78, 102, 'Zara', 'L'],
    [155, 48, 80, 62, 86, 'Zara', 'XS'],
    
    # H&M (Similar to Zara, European sizing)
    [158, 49, 81, 63, 87, 'H&M', 'XS'],
    [163, 54, 85, 67, 91, 'H&M', 'S'],
    [168, 59, 89, 71, 95, 'H&M', 'M'],
    [173, 64, 93, 75, 99, 'H&M', 'L'],
    [178, 69, 97, 79, 103, 'H&M', 'XL'],
    [161, 51, 83, 65, 89, 'H&M', 'S'],
    [166, 57, 87, 69, 93, 'H&M', 'M'],
    [171, 61, 91, 73, 97, 'H&M', 'M'],
    [176, 67, 95, 77, 101, 'H&M', 'L'],
    [153, 47, 79, 61, 85, 'H&M', 'XS'],
    
    # Uniqlo (Asian sizing - runs smaller)
    [160, 50, 80, 62, 86, 'Uniqlo', 'M'],
    [165, 55, 84, 66, 90, 'Uniqlo', 'L'],
    [170, 60, 88, 70, 94, 'Uniqlo', 'XL'],
    [175, 65, 92, 74, 98, 'Uniqlo', 'XXL'],
    [155, 48, 78, 60, 84, 'Uniqlo', 'S'],
    [162, 52, 82, 64, 88, 'Uniqlo', 'M'],
    [167, 57, 86, 68, 92, 'Uniqlo', 'L'],
    [172, 62, 90, 72, 96, 'Uniqlo', 'XL'],
    [158, 49, 79, 61, 85, 'Uniqlo', 'S'],
    [174, 64, 91, 73, 97, 'Uniqlo', 'XL'],
    
    # Gap (American sizing - tends to run larger)
    [160, 50, 84, 66, 90, 'Gap', 'S'],
    [165, 55, 88, 70, 94, 'Gap', 'S'],
    [170, 60, 92, 74, 98, 'Gap', 'M'],
    [175, 65, 96, 78, 102, 'Gap', 'M'],
    [180, 70, 100, 82, 106, 'Gap', 'L'],
    [162, 52, 86, 68, 92, 'Gap', 'S'],
    [168, 58, 90, 72, 96, 'Gap', 'M'],
    [172, 62, 94, 76, 100, 'Gap', 'M'],
    [178, 68, 98, 80, 104, 'Gap', 'L'],
    [155, 48, 82, 64, 88, 'Gap', 'XS'],
    
    # Old Navy (American sizing - runs large)
    [160, 50, 86, 68, 92, 'Old Navy', 'XS'],
    [165, 55, 90, 72, 96, 'Old Navy', 'S'],
    [170, 60, 94, 76, 100, 'Old Navy', 'S'],
    [175, 65, 98, 80, 104, 'Old Navy', 'M'],
    [180, 70, 102, 84, 108, 'Old Navy', 'M'],
    [162, 52, 88, 70, 94, 'Old Navy', 'XS'],
    [168, 58, 92, 74, 98, 'Old Navy', 'S'],
    [172, 62, 96, 78, 102, 'Old Navy', 'S'],
    [178, 68, 100, 82, 106, 'Old Navy', 'M'],
    [155, 48, 84, 66, 90, 'Old Navy', 'XS'],
    
    # Shein (Asian sizing - runs very small)
    [160, 50, 78, 60, 84, 'Shein', 'L'],
    [165, 55, 82, 64, 88, 'Shein', 'XL'],
    [170, 60, 86, 68, 92, 'Shein', 'XXL'],
    [155, 48, 76, 58, 82, 'Shein', 'M'],
    [162, 52, 80, 62, 86, 'Shein', 'L'],
    [167, 57, 84, 66, 90, 'Shein', 'XL'],
    [172, 62, 88, 70, 94, 'Shein', 'XXL'],
    [158, 49, 77, 59, 83, 'Shein', 'M'],
    [174, 64, 90, 72, 96, 'Shein', 'XXL'],
    [153, 47, 75, 57, 81, 'Shein', 'M'],
]

def create_enhanced_model():
    # Convert to DataFrame
    df = pd.DataFrame(training_data, columns=['height', 'weight', 'chest', 'waist', 'hips', 'brand', 'size'])
    
    # Encode brand names
    brand_encoder = LabelEncoder()
    df['brand_encoded'] = brand_encoder.fit_transform(df['brand'])
    
    # Prepare features (include brand as a feature)
    X = df[['height', 'weight', 'chest', 'waist', 'hips', 'brand_encoded']].values
    y = df['size'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest (better for this type of data than Decision Tree)
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and encoders
    joblib.dump(model, 'enhanced_model.pkl')
    joblib.dump(brand_encoder, 'brand_encoder.pkl')
    
    # Save brand mapping for reference
    brand_mapping = {brand: idx for idx, brand in enumerate(brand_encoder.classes_)}
    with open('brand_mapping.json', 'w') as f:
        json.dump(brand_mapping, f)
    
    print("\nModel saved as 'enhanced_model.pkl'")
    print("Brand encoder saved as 'brand_encoder.pkl'")
    print("Brand mapping saved as 'brand_mapping.json'")
    
    # Display feature importances
    feature_names = ['height', 'weight', 'chest', 'waist', 'hips', 'brand']
    importances = model.feature_importances_
    print("\nFeature Importances:")
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.3f}")
    
    return model, brand_encoder

if __name__ == "__main__":
    model, brand_encoder = create_enhanced_model()
    
    # Test the model with a sample prediction
    print("\n" + "="*50)
    print("Testing model with sample data:")
    
    # Example: 170cm, 60kg person looking at Zara
    test_brand = "Zara"
    test_features = np.array([[170, 60, 90, 72, 96, brand_encoder.transform([test_brand])[0]]])
    prediction = model.predict(test_features)[0]
    probabilities = model.predict_proba(test_features)[0]
    
    print(f"Test case: 170cm, 60kg, looking at {test_brand}")
    print(f"Predicted size: {prediction}")
    print("Size probabilities:")
    for size, prob in zip(model.classes_, probabilities):
        print(f"  {size}: {prob:.3f}")