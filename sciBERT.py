import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import pickle
import json

def preprocess_function(examples):
    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the text field
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512,           # adjust if needed
        return_tensors='pt'
    )


def predict(model, dataset):
    model.eval()
    predictions = []
    
    for i in range(len(dataset)):
        batch = dataset[i]
        
        # Only keep the keys the model actually needs
        inputs = {
            'input_ids': batch['input_ids'].unsqueeze(0).to(model.device),
            'attention_mask': batch['attention_mask'].unsqueeze(0).to(model.device)
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()
        predictions.append(pred)
    
    return predictions


def get_results():
    label_mapping = {
        'general-url': 0,
        'third-party-dataset': 1,
        'author-provided-dataset': 2,
        'third-party-software': 3,
        'author-provided-software': 4,
        'project': 5
    }
    
    model_name = "allenai/scibert_scivocab_uncased"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_mapping)
    )
    
    # Load saved weights
    model_file_path = 'saved_weights/sciBERT_weights.pkl'  # adjust path if needed
    with open(model_file_path, 'rb') as f:
        state_dict = pickle.load(f)
    model.load_state_dict(state_dict)
    
    # ────────────────────────────────────────────────────────────────
    # CHANGED: Load the full JSON test set instead of the CSV
    test_json_path = 'dataset/OADS_Test.json'   # ← same file SCL is using
    
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    test_df = pd.DataFrame(test_data)
    
    # Standardize column name (some files use 'Text', most use 'text')
    if 'text' not in test_df.columns and 'Text' in test_df.columns:
        test_df = test_df.rename(columns={'Text': 'text'})
    
    if 'text' not in test_df.columns:
        raise ValueError("No 'text' column found in the loaded JSON data")
    
    print(f"[SciBERT] Loaded {len(test_df)} examples from {test_json_path}")
    if len(test_df) > 0:
        print(f"    First text preview: {test_df['text'].iloc[0][:120]}...")
    # ────────────────────────────────────────────────────────────────

    # Convert to Hugging Face Dataset
    test_dataset = Dataset.from_pandas(test_df)
    
    # Apply tokenization
    test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)
    
    # Set format for PyTorch
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
    
    # Run prediction
    print("[SciBERT] Starting prediction...")
    predictions = predict(model, test_dataset)
    
    print(f"[SciBERT] Generated {len(predictions)} predictions")
    
    return predictions


if __name__ == '__main__':
    preds = get_results()
    print("Sample predictions (first 10):", preds[:10])