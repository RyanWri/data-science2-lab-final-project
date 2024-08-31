import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import yaml

def load_config(config_path='config.yaml'):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def prepare_data_for_modeling(merged_medications, config):
    """Prepare data for modeling by encoding categorical variables and splitting into training and testing sets."""
    print("Preparing data for modeling...")
    if 're_hospitalized' not in merged_medications.columns:
        if 'Admission_Entry_Date' in merged_medications.columns:
            merged_medications['re_hospitalized'] = merged_medications['Admission_Entry_Date'].duplicated(keep=False).astype(int)
        else:
            print("Error: 'Admission_Entry_Date' column not found.")
            return None, None, None, None

    features = merged_medications[['age', 'Gender', 'BMI', 'Drug']].copy()
    target = merged_medications['re_hospitalized']

    features = pd.get_dummies(features, columns=['Gender', 'Drug'], drop_first=True)
    features.fillna(0, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    print("Data prepared for modeling.")
    return X_train, X_test, y_train, y_test

def build_and_train_model(X_train, y_train, X_test, y_test, config):
    """Build and train the neural network model based on the configuration."""
    print("Building model...")
    model = tf.keras.Sequential()
    
    for layer in config['model']['layers']:
        if layer['type'] == 'Dense':
            model.add(tf.keras.layers.Dense(units=layer['units'], activation=layer['activation']))
        elif layer['type'] == 'Dropout':
            model.add(tf.keras.layers.Dropout(rate=layer['rate']))

    model.compile(optimizer=config['model']['optimizer'],
                  loss=config['model']['loss'],
                  metrics=config['model']['metrics'])

    print("Starting model training...")
    history = model.fit(X_train, y_train, epochs=config['training']['epochs'],
                        batch_size=config['training']['batch_size'],
                        validation_data=(X_test, y_test),
                        verbose=1)
    print("Model training completed.")

    # Plot loss per epoch
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pictures/loss_per_epoch.png')
    plt.close()

    # Plot accuracy per epoch
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pictures/accuracy_per_epoch.png')
    plt.close()

    return model

def plot_common_medications(merged_medications):
    """Plot and save the 20 most common medications."""
    plt.figure(figsize=(10, 6))
    common_medications = merged_medications['Drug'].value_counts().head(20)
    common_medications.plot(kind='bar', color='skyblue')
    plt.title('Top 20 Most Common Medications')
    plt.xlabel('Medication')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('pictures/top_20_common_medications.png')
    plt.close()

if __name__ == "__main__":
    print("Loading configuration...")
    config = load_config('config.yaml')  # Use relative path for config file

    print("Loading processed medication data...")
    merged_medications = pd.read_csv('medications_data/merged_medications.csv')  # Use relative path for data file

    print("Columns in merged_medications:", merged_medications.columns)

    X_train, X_test, y_train, y_test = prepare_data_for_modeling(merged_medications, config)

    if X_train is not None:
        model = build_and_train_model(X_train, y_train, X_test, y_test, config)
        model.save('rehospitalization_model.h5')  # Use relative path for saving the model
        print("Model training completed and saved.")

        # Plot the top 20 most common medications
        plot_common_medications(merged_medications)
        print("Top 20 common medications chart saved.")

    else:
        print("Data preparation failed. Model training aborted.")