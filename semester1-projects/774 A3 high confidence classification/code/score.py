import pandas as pd
import os

def calculate_score(true_labels: pd.Series, predicted_labels: pd.Series, accuracy_threshold: float = 0.99, gamma: float = 5.0):
    """
    Calculate the overall score based on correct classifications for high and low accuracy classes.
    Track the number of correct, incorrect, and skipped predictions.

    Parameters:
        - true_labels: A pandas Series containing the true class labels.
        - predicted_labels: A pandas Series containing the predicted class labels.
        - accuracy_threshold: Threshold for class accuracy.
        - gamma: Weighting factor for low accuracy classifications.

    Returns:
        - A tuple (final_score, total_correct, total_incorrect, total_skipped)
    """
    
    # Combine the true labels and predicted labels into a DataFrame
    data = pd.DataFrame({'True_label': true_labels, 'Predicted_label': predicted_labels})
    
    # Count skipped predictions
    total_skipped = (data['Predicted_label'] == -1).sum()
    
    # Filter out skipped predictions
    filtered_df = data[data['Predicted_label'] != -1]
    
    all_classes = list(range(100))  # Assuming CIFAR-100 dataset with 100 classes
    sum_of_correctly_classified_high_accuracy = 0
    sum_of_correctly_classified_low_accuracy = 0
    
    # Initialize counts for correct and incorrect predictions
    total_correct = 0
    total_incorrect = 0
    
    # Calculate accuracy per class
    accuracy_per_class = {}
    grouped = filtered_df.groupby('Predicted_label')
    
    for name, group in grouped:
        accuracy = (group['True_label'] == group['Predicted_label']).sum() / len(group)
        accuracy_per_class[name] = accuracy
        total_correct += (group['True_label'] == group['Predicted_label']).sum()
        total_incorrect += (group['True_label'] != group['Predicted_label']).sum()
    
    for cls in all_classes:
        total = len(filtered_df[filtered_df['Predicted_label'] == cls])
        class_accuracy = accuracy_per_class.get(cls, 0.0)
        
        if class_accuracy >= accuracy_threshold:
            sum_of_correctly_classified_high_accuracy += total
        else:
            sum_of_correctly_classified_low_accuracy += total

    # Calculate final score
    final_score = sum_of_correctly_classified_high_accuracy - gamma * sum_of_correctly_classified_low_accuracy

    return final_score, total_correct, total_incorrect, total_skipped

def evaluate_in_two_sets(test_info_path: str, prediction_file: str):
    # Load the test_info.csv containing ID and True_label
    test_info = pd.read_csv(test_info_path)
    
    # Load the predicted labels from the submission file
    predictions = pd.read_csv(prediction_file)
    
    # Merge the predictions with the true labels using the ID column
    merged_data = pd.merge(test_info, predictions, on='ID')
    
    # Split into two sets of 10,000
    first_half = merged_data[:10000]
    second_half = merged_data[10000:20000]
    
    # Calculate the score for the first set
    score_first_half, correct_first_half, incorrect_first_half, skipped_first_half = calculate_score(first_half['True_label'], first_half['Predicted_label'])
    
    # Calculate the score for the second set
    score_second_half, correct_second_half, incorrect_second_half, skipped_second_half = calculate_score(second_half['True_label'], second_half['Predicted_label'])
    
    # Store the results for both sets
    batch_1 = {
        'total': 10000,
        'correct': correct_first_half,
        'incorrect': incorrect_first_half,
        'skipped': skipped_first_half,
        'score': score_first_half
    }
    
    batch_2 = {
        'total': 10000,
        'correct': correct_second_half,
        'incorrect': incorrect_second_half,
        'skipped': skipped_second_half,
        'score': score_second_half
    }
    
    return batch_1, batch_2

def evaluate_all_files_in_folder(folder_path: str, test_info_path: str, output_excel_file: str):
    # List all CSV files in the given folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Initialize a list to store the results
    results = []
    
    # Loop through each file and calculate the score
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        print(f"\nEvaluating {csv_file}")
        
        # Evaluate the batches
        batch_1, batch_2 = evaluate_in_two_sets(test_info_path, file_path)
        
        # Append the results to the list
        results.append({
            'file_name': csv_file,
            'batch_1_total': batch_1['total'],
            'batch_1_correct': batch_1['correct'],
            'batch_1_incorrect': batch_1['incorrect'],
            'batch_1_skipped': batch_1['skipped'],
            'batch_1_score': batch_1['score'],
            'batch_2_total': batch_2['total'],
            'batch_2_correct': batch_2['correct'],
            'batch_2_incorrect': batch_2['incorrect'],
            'batch_2_skipped': batch_2['skipped'],
            'batch_2_score': batch_2['score'],
        })
    
    # Convert the results to a DataFrame
    df_results = pd.DataFrame(results)
    
    # Save the DataFrame to an Excel file
    df_results.to_excel(output_excel_file, index=False)
    print(f"Results saved to {output_excel_file}")

# Example usage
folder_path = 'predictions/'  # Path to the folder containing prediction CSV files
test_info_path = 'test_info.csv'  # Path to test_info.csv containing the true labels
output_excel_file = 'evaluation_results.xlsx'  # Path to the output Excel file

evaluate_all_files_in_folder(folder_path, test_info_path, output_excel_file)

