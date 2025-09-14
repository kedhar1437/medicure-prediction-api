import pandas as pd
import numpy as np

# Original + Additional common diseases and their symptoms
disease_symptoms = {
    'Common Cold': ['runny_nose', 'sneezing', 'sore_throat', 'cough', 'congestion', 'mild_fatigue'],
    'Influenza': ['fever', 'body_aches', 'fatigue', 'headache', 'chills', 'dry_cough'],
    'Migraine': ['severe_headache', 'nausea', 'light_sensitivity', 'sound_sensitivity', 'aura'],
    'Stomach Flu': ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain', 'low_grade_fever'],
    'Allergies': ['sneezing', 'itchy_eyes', 'runny_nose', 'congestion', 'itchy_throat'],
    'Pneumonia': ['high_fever', 'cough_with_phlegm', 'shortness_of_breath', 'chest_pain', 'fatigue'],
    'Bronchitis': ['persistent_cough', 'mucus_production', 'wheezing', 'chest_discomfort', 'low_fever'],
    'Sinusitis': ['facial_pain', 'nasal_congestion', 'thick_nasal_discharge', 'reduced_smell', 'headache'],
    'Strep Throat': ['severe_sore_throat', 'painful_swallowing', 'fever', 'swollen_lymph_nodes', 'white_patches'],
    'COVID-19': ['fever', 'dry_cough', 'fatigue', 'loss_of_taste', 'loss_of_smell', 'shortness_of_breath'],

    # NEWLY ADDED DISEASES
    'Asthma': ['shortness_of_breath', 'wheezing', 'chest_tightness', 'cough'],
    'Diabetes': ['frequent_urination', 'excessive_thirst', 'fatigue', 'blurred_vision', 'slow_healing'],
    'Hypertension': ['headache', 'dizziness', 'blurred_vision', 'chest_pain'],
    'Depression': ['sadness', 'fatigue', 'difficulty_sleeping', 'loss_of_appetite', 'irritability'],
    'Anxiety': ['restlessness', 'rapid_heartbeat', 'sweating', 'difficulty_concentrating'],
    'Tuberculosis': ['persistent_cough', 'night_sweats', 'weight_loss', 'fever', 'fatigue'],
    'Dengue': ['high_fever', 'rash', 'muscle_pain', 'joint_pain', 'nausea'],
    'Malaria': ['fever', 'chills', 'sweating', 'headache', 'nausea'],
    'Typhoid': ['fever', 'abdominal_pain', 'fatigue', 'headache', 'diarrhea'],
    'Chickenpox': ['itchy_rash', 'fever', 'fatigue', 'loss_of_appetite', 'body_aches']
}

# Additional general symptoms
additional_symptoms = [
    'mild_fever', 'dizziness', 'loss_of_appetite', 'muscle_weakness',
    'sweating', 'irritability', 'difficulty_sleeping', 'blurred_vision',
    'restlessness', 'rapid_heartbeat', 'chest_pain', 'itchy_throat',
    'slow_healing', 'weight_loss', 'itchy_rash', 'painful_swallowing'
]

# Set of all symptoms
all_symptoms = set()
for symptoms in disease_symptoms.values():
    all_symptoms.update(symptoms)
all_symptoms.update(additional_symptoms)

# Generate synthetic data
records = []
np.random.seed(42)

# Generate 400 records per disease (8000 total)
for disease, primary_symptoms in disease_symptoms.items():
    for _ in range(400):
        record = {'disease': disease}

        # Set primary symptoms (high probability)
        for symptom in primary_symptoms:
            record[symptom] = np.random.choice([0, 1], p=[0.1, 0.9])  # 90% chance

        # Set secondary symptoms (low probability)
        for symptom in all_symptoms - set(primary_symptoms):
            prob = 0.05
            if symptom in ['fever', 'fatigue', 'headache', 'nausea']:
                prob = 0.2
            record[symptom] = np.random.choice([0, 1], p=[1 - prob, prob])

        records.append(record)

# Create DataFrame
df = pd.DataFrame(records)

# Flip 5% symptom values for noise
for col in df.columns[1:]:
    if np.random.rand() < 0.05:
        df[col] = df[col].sample(frac=1).values

# Save to CSV
df.to_csv("disease_symptoms_extended.csv", index=False)
print(f"Created dataset with {len(df)} records and {len(df.columns)-1} symptoms.")

