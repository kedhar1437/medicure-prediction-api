import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import classification_report
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "disease_symptoms_extended.csv"
MODEL_PATH = BASE_DIR / "disease_model_enhanced.joblib"

# Load enhanced dataset
print("Loading dataset from:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

# Prepare data
X = df.drop('disease', axis=1)
y = df['disease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
print("Test Set Evaluation:")
print(classification_report(y_test, model.predict(X_test)))

# Save model with metadata
joblib.dump({
    'model': model,
    'features': X.columns.tolist(),
    'classes': model.classes_.tolist(),
    'symptom_descriptions': {col: col.replace("_", " ") for col in X.columns},
    'disease_info': {
        'Common Cold': 'Viral infection of the upper respiratory tract',
        'Influenza': 'Contagious respiratory illness caused by influenza viruses',
        'Migraine': 'Neurological condition causing intense headaches and sensitivity to light/sound',
        'Stomach Flu': 'Viral gastroenteritis causing vomiting, diarrhea, and cramps',
        'Allergies': 'Immune system reaction to allergens like pollen, dust, etc.',
        'Pneumonia': 'Lung infection that inflames air sacs and may cause cough with phlegm',
        'Bronchitis': 'Inflammation of bronchial tubes with persistent cough',
        'Sinusitis': 'Inflammation of the sinuses, often due to infection or allergies',
        'Strep Throat': 'Bacterial throat infection with pain and fever',
        'COVID-19': 'Infectious disease caused by the SARS-CoV-2 virus',
        'Asthma': 'Chronic respiratory condition causing wheezing and breathlessness',
        'Diabetes': 'Metabolic disease causing high blood sugar levels',
        'Hypertension': 'High blood pressure, often asymptomatic but risky for heart/brain',
        'Depression': 'Mental health disorder with persistent sadness and fatigue',
        'Anxiety': 'Excessive worry and tension often accompanied by physical symptoms',
        'Tuberculosis': 'Bacterial infection primarily affecting the lungs',
        'Dengue': 'Mosquito-borne viral infection causing fever, rash, and muscle pain',
        'Malaria': 'Mosquito-borne disease caused by Plasmodium parasites',
        'Typhoid': 'Bacterial infection causing fever, abdominal pain, and weakness',
        'Chickenpox': 'Contagious viral infection causing itchy rash and fever'
    }
}, MODEL_PATH)

print(f"✅ Model trained and saved successfully at {MODEL_PATH}")
