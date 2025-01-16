from datasets import load_dataset
import pandas as pd

def analyze_xray_labels(dataset_name="danjacobellis/chexpert"):
    # Load the dataset
    dataset = load_dataset(dataset_name)
    
    # Convert to pandas DataFrame for easier analysis
    df = pd.DataFrame(dataset['train'])
    
    # List of all possible label columns (excluding metadata columns)
    label_columns = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices'
    ]
    
    # Initialize results dictionary
    label_analysis = {}
    
    # Analyze each label column
    for col in label_columns:
        if col in df.columns:
            value_counts = df[col].value_counts().to_dict()
            label_analysis[col] = {
                'unique_classes': sorted(list(value_counts.keys())),
                'class_counts': value_counts
            }
    
    # Analyze view types and demographic columns
    metadata_cols = ['Frontal/Lateral', 'AP/PA', 'Sex']
    for col in metadata_cols:
        if col in df.columns:
            value_counts = df[col].value_counts().to_dict()
            label_analysis[col] = {
                'unique_classes': sorted(list(value_counts.keys())),
                'class_counts': value_counts
            }
    
    return label_analysis

# Run the analysis
results = analyze_xray_labels()

# Print results in a clean format
for label, info in results.items():
    print(f"\n{label}:")
    print("Unique classes:", info['unique_classes'])
    print("Class counts:")
    for class_val, count in info['class_counts'].items():
        print(f"  Class {class_val}: {count} samples")