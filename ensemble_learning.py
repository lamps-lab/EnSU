import torch
import json
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.metrics import classification_report
from scl import Instructor
from config import get_config
from sciBERT import get_results

# Load the model from the specified path
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    return model

# Get predictions from the model
def get_predictions(model, input_data):
    with torch.no_grad():
        tensor_inputs = torch.tensor(input_data).float()  # Convert input data to tensor
        model_outputs = model(tensor_inputs)
        probabilities = torch.softmax(model_outputs, dim=1)  # Apply softmax to get probabilities
        predicted_classes = torch.argmax(probabilities, dim=1)
    return predicted_classes.numpy()  # Return predictions as a numpy array

# Perform majority voting for ensemble predictions with handling for all-different predictions
def majority_voting(*prediction_sets):
    stacked_predictions = np.vstack(prediction_sets)  # Stack all predictions
    final_predictions = []

    for row in stacked_predictions.T:  # Iterate over each set of predictions for a sample
        row_counts = Counter(row)
        if len(row_counts) == 2:  # If all predictions are different
            final_predictions.append(row[1])  # Choose the third prediction (mapped_labels)
        else:
            majority_vote = row_counts.most_common(1)[0][0]  # Otherwise, get the majority vote
            final_predictions.append(majority_vote)

    return np.array(final_predictions)

# Load data from a JSON file

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:  # Add encoding='utf-8'
        return json.load(file)
#def load_data(file_path):
#    with open(file_path, 'r') as file:
#        return json.load(file)  # Return the loaded data directly

# Main function to execute the workflow
def main():
    # Load training and test data from JSON files
    training_data = load_data('dataset/OADS_Train.json')  # Training input features
    testing_data = load_data('dataset/OADS_Test.json')    # Testing input features

    args, logger = get_config()
    instructor = Instructor(args, logger)


    predictions_csv_path = 'D:/Server/OADS-HT25-D29E-V2/OADS-HT25/dataset/predictions-mapped.csv'  # Path to your CSV file with predictions

    # Get predictions using SCIBERT the classifiers
    sciBERT_predictions = get_results()
    #true_labels, scl_predictions = instructor.run()
    true_labels, scl_predictions = instructor.run()

    # Define label mapping
    label_mapping = {
        'general-url': 0,
        'third-party-dataset': 1,
        'author-provided-dataset': 2,
        'third-party-software': 3,
        'author-provided-software': 4,
        'project': 5
    }

    # Read the CSV file with predictions
    predictions_df = pd.read_csv(predictions_csv_path)

    # Map the labels in the DataFrame
    bertGCN_predictions = predictions_df['label'].map(label_mapping).tolist()

    # Combine predictions from different classifiers (including mapped_labels)
    #final_predictions = majority_voting(scl_predictions, predictions_df['predicted_label'])
    final_predictions = majority_voting(scl_predictions, bertGCN_predictions, sciBERT_predictions)

    if true_labels is not None:
        print(classification_report(true_labels, final_predictions))
    else:
        print("Final ensemble predictions generated (no true labels available).")
    # Optionally save predictions
    np.save("final_ensemble_predictions.npy", final_predictions)
    print("Saved predictions to final_ensemble_predictions.npy")
    # Print classification reports
    
#    print("Classification Report:")
#    print(classification_report(true_labels, final_predictions))
    print(final_predictions)
if __name__ == '__main__':
    main()

