import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import pickle
import ijson  # pip install ijson   ← install this if not already present
from tqdm import tqdm

def preprocess_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )


def predict(model, dataset):
    """Predict in smaller batches to save memory"""
    model.eval()
    predictions = []
    
    batch_size = 32  # adjust lower if still OOM
    for i in tqdm(range(0, len(dataset), batch_size), desc="Predicting"):
        batch = dataset[i:i + batch_size]
        
        inputs = {
            'input_ids': batch['input_ids'].to(model.device),
            'attention_mask': batch['attention_mask'].to(model.device)
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        predictions.extend(preds)
    
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_mapping)
    )
    
    # Load saved weights
    model_file_path = 'saved_weights/sciBERT_weights.pkl'
    with open(model_file_path, 'rb') as f:
        state_dict = pickle.load(f)
    model.load_state_dict(state_dict)
    
    print("Model loaded successfully.")
    
    # ── STREAMING LOAD for very large JSON ──────────────────────────────────
    test_json_path = 'dataset/OADS_Test.json'
    
    print(f"[SciBERT] Streaming large JSON file: {test_json_path}")
    
    texts = []
    
    with open(test_json_path, 'rb') as f:  # binary mode for ijson
        objects = ijson.items(f, 'item')   # iterates over each object in the array
        
        for obj in tqdm(objects, desc="Reading texts"):
            # Look for the text field (handles 'Text' or 'text')
            text = obj.get('Text') or obj.get('text') or obj.get('sentence')
            if text and isinstance(text, str) and text.strip():
                texts.append(text)
    
    if not texts:
        raise ValueError("No valid text entries found in the JSON file")
    
    print(f"[SciBERT] Successfully extracted {len(texts)} text entries (streaming mode)")
    
    # Create minimal DataFrame → only keeps the text column
    test_df = pd.DataFrame({'text': texts})
    
    if len(test_df) > 0:
        preview = test_df['text'].iloc[0][:120].replace('\n', ' ').strip()
        print(f"    First text preview: {preview}...")
    # ────────────────────────────────────────────────────────────────────────

    # Convert to Hugging Face Dataset
    test_dataset = Dataset.from_pandas(test_df)
    
    # Tokenize in batches + remove original text column to save memory
    print("[SciBERT] Tokenizing dataset...")
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=500,               # smaller batch size → less memory spike
        remove_columns=['text']       # discard raw text immediately
    )
    
    # Set PyTorch format
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
    
    print("[SciBERT] Starting prediction...")
    predictions = predict(model, test_dataset)
    
    print(f"[SciBERT] Generated {len(predictions)} predictions")
    
    return predictions


if __name__ == '__main__':
    try:
        preds = get_results()
        print("\nSample predictions (first 10):", preds[:10])
    except Exception as e:
        print("Error during execution:")
        print(e)