# 🔧 STEP 0: Install Required Libraries
# If you haven't already, run this in your terminal or command prompt:
# pip install numpy pandas matplotlib scikit-learn

# 📦 Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- Configuration ---
# 🚩 This should be your newly created CSV file with extracted features and emotions.
PROCESSED_FEATURES_CSV_PATH = "eeg_features_emotions.csv"

# Define the feature columns that your model will be trained on.
# These names MUST EXACTLY MATCH the column names in your PROCESSED_FEATURES_CSV_PATH.
EXPECTED_FEATURE_COLUMNS = ['delta', 'theta', 'alpha', 'beta', 'gamma']

print("*********************************************************************")
print("*** EEG Emotion Recognition - Machine Learning Script (Demonstration) ***")
print("*********************************************************************\n")

def train_model_for_demonstration(csv_path, feature_column_names):
    """
    Loads data, trains a model on ALL available data (for demonstration with very small datasets),
    and returns the trained model. Evaluation is skipped due to insufficient data.
    """
    print("\n--- LOADING AND PREPARING DATA ---")
    try:
        data = pd.read_csv(csv_path)
        print(f"✅ Successfully loaded '{csv_path}'")
    except FileNotFoundError:
        print(f"❌ ERROR: The file '{csv_path}' was not found.")
        print("Please make sure this is your PROCESSED feature CSV file and it's in the correct path.")
        return None

    print("\n📄 Sample of Loaded Data:")
    print(data.head())
    # print("\n📊 Data Info:") # Can be verbose for presentation, enable if needed
    # data.info()

    # Validate essential columns
    if 'emotion' not in data.columns:
        print(f"\n❌ ERROR: The CSV file must contain an 'emotion' column for labels.")
        return None
    for feature_col in feature_column_names:
        if feature_col not in data.columns:
            print(f"\n❌ ERROR: Expected feature column '{feature_col}' not found in '{csv_path}'.")
            print(f"   Available columns: {data.columns.tolist()}")
            return None

    X = data[feature_column_names]
    y = data['emotion']

    print(f"\n🧠 Features (X) for training (first 5 rows):\n{X.head()}")
    print(f"\n🎯 Labels (y) for training (first 5 rows):\n{y.head()}")
    print(f"\nTotal number of data samples: {len(data)}")

    if len(data) < 5:
        print("⚠️ WARNING: Dataset is very small! Model will be trained on all available data for demonstration purposes only.")
        print("   Evaluation (like accuracy scores) will be skipped as it's not meaningful with so few samples.")
        print("   For a real project, you need many more data samples.")
        
        if len(data) < 2 or y.nunique() < 2 : # Need at least 2 samples and 2 unique classes to train
            print("❌ ERROR: Not enough data or not enough unique emotion classes to train even a demonstration model.")
            print(f"   Samples: {len(data)}, Unique Emotion Classes: {y.nunique()}")
            return None
            
        # Train on all data
        X_train, y_train = X, y
        model_trained_on_all_data = True
    else:
        # If you had more data (e.g., > 10-20 samples), you would split it:
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)
        # print(f"\nData split into: {len(X_train)} training samples and {len(X_test)} testing samples.")
        # For now, with very few samples, we stick to training on all data.
        # This part is just for illustration if you expand your dataset later.
        print("Proceeding to train on all available data due to small dataset size, as configured.")
        X_train, y_train = X, y
        model_trained_on_all_data = True


    print("\n--- MODEL TRAINING ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    try:
        model.fit(X_train, y_train)
        print("✅ Model trained successfully!")
    except Exception as e:
        print(f"❌ ERROR during model training: {e}")
        print("   This can happen if there are issues with the data (e.g., too few samples per class).")
        return None

    if model_trained_on_all_data:
        print("   Note: Model was trained on ALL available data. No separate test set was used for evaluation.")
    
    # --- MODEL EVALUATION (SKIPPED FOR VERY SMALL DATA) ---
    # If you had enough data for X_test, y_test, you would do:
    # print("\n--- MODEL EVALUATION ---")
    # y_pred = model.predict(X_test)
    # print("📊 Classification Report (on Test Set):")
    # print(classification_report(y_test, y_pred, zero_division=0))
    #
    # if hasattr(model, 'feature_importances_'):
    #     print("\n--- FEATURE IMPORTANCE ---")
    #     importances = model.feature_importances_
    #     plt.figure(figsize=(10, 5))
    #     plt.bar(feature_column_names, importances, color='skyblue')
    #     plt.xlabel("EEG Features")
    #     plt.ylabel("Importance Score")
    #     plt.title("Feature Importance in Emotion Prediction (Demonstration)")
    #     plt.xticks(rotation=45, ha="right")
    #     plt.tight_layout()
    #     print("ℹ️ Displaying Feature Importance plot...")
    #     plt.show()

    return model

def predict_emotion_from_input(model, feature_column_names):
    """
    Takes live input for EEG features and predicts emotion.
    """
    if model is None:
        print("\n❌ Model is not available. Cannot make predictions.")
        return

    print("\n\n--- INTERACTIVE EMOTION PREDICTION ---")
    print("Instructions: Enter the EEG feature values when prompted.")
    print(f"You will need to enter numerical values for: {', '.join(feature_column_names)}")
    print("Example values from your file:")
    print("  Happy: delta=741, theta=16.8, alpha=6.6, beta=6.8, gamma=5.0")
    print("  Sad:   delta=544, theta=18.1, alpha=8.5, beta=8.0, gamma=7.6")
    print("Type 'quit' at any prompt to exit.")

    try:
        input_features = []
        for feature_name in feature_column_names:
            while True:
                try:
                    value_str = input(f"➡️ Enter value for {feature_name}: ").strip().lower()
                    if value_str == 'quit':
                        print("\nExiting interactive prediction...")
                        return False # Signal to exit loop
                    value = float(value_str)
                    input_features.append(value)
                    break
                except ValueError:
                    print("   ⚠️ Invalid input. Please enter a numerical value (e.g., 15.5) or 'quit'.")
        
        input_array = np.array([input_features]) # Reshape for single prediction
        
        predicted_emotion = model.predict(input_array)
        predicted_probabilities = model.predict_proba(input_array)
        
        print(f"\n💡 PREDICTED EMOTION: {predicted_emotion[0].upper()} 🎉")

        print("\nConfidence Scores (Probabilities per class):")
        classes = model.classes_
        for i, emotion_class in enumerate(classes):
            print(f"  - {emotion_class}: {predicted_probabilities[0][i]*100:.2f}%")
        
        print("-----------------------------------------")
        return True # Signal to continue loop

    except KeyboardInterrupt:
        print("\n🛑 Interactive prediction stopped by user.")
        return False
    except Exception as e:
        print(f"\n❌ An error occurred during prediction: {e}")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    # Train the model using the processed feature CSV
    trained_model = train_model_for_demonstration(PROCESSED_FEATURES_CSV_PATH, EXPECTED_FEATURE_COLUMNS)
    
    if trained_model:
        print("\n✅ Demonstration model is ready for interactive predictions.")
        print("   (Remember: This model is trained on very limited data and is for demonstration only!)")
        while True:
            continue_predicting = predict_emotion_from_input(trained_model, EXPECTED_FEATURE_COLUMNS)
            if not continue_predicting:
                break
    else:
        print("\n❌ Model could not be trained. Please check the data and previous error messages.")
        print("   For a successful run, ensure 'eeg_features_emotions.csv' exists and has valid data.")

    print("\n*********************************************************************")
    print("*** End of EEG Emotion Recognition Demonstration Script             ***")
    print("*********************************************************************")