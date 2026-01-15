import os
import json
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader


# class MyDataset(Dataset):

#     def __init__(self, raw_data, label_dict, tokenizer, model_name, method):
        
#         label_list = list(label_dict.keys()) if method not in ['ce', 'scl'] else []
#         sep_token = ['[SEP]'] if model_name == 'bert' else ['</s>']
#         dataset = list()
#         for data in raw_data:
#             tokens = data['Text'].lower().split(' ')
#             print(data['Text'])
#             label_id = label_dict[data['label']]
#             dataset.append((label_list + sep_token + tokens, label_id))
#         self._dataset = dataset

#     def __getitem__(self, index):
#         return self._dataset[index]

#     def __len__(self):
#         return len(self._dataset)
class MyDataset(Dataset):
    def __init__(self, data, label_dict, tokenizer, model_name, method):
        self.data = data
        self.label_dict = label_dict
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.method = method
        
        # Preprocess and cache tokenized data
        self.examples = []
        for item in data:
            text = item.get('text', item.get('Text', ''))  # Handle both key names
            label_id = None
            if 'label' in item:
                label_id = label_dict[item['label']]
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,  # Adjust as needed
                return_tensors='pt'
            )
            
            self.examples.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': label_id  # Can be None for test data
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
    
# def my_collate(batch):
#     # batch is a list of dicts: [{'input_ids': tensor, 'attention_mask': tensor, 'label': int or None}, ...]
    
#     input_ids = torch.stack([item['input_ids'] for item in batch])
#     attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
#     # Labels may be None for test data, handle accordingly
#     labels = [item['label'] for item in batch]
#     if labels[0] is not None:
#         labels = torch.tensor(labels, dtype=torch.long)
#     else:
#         labels = None  # or torch.tensor([-1] * len(batch)) if your model expects a tensor
    
#     return {
#         'input_ids': input_ids,
#         'attention_mask': attention_mask,
#     }, labels

def my_collate(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    labels = [item['label'] for item in batch]
    if labels[0] is not None:
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        labels = None  # or handle as needed for test set
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }, labels
#def my_collate(batch, tokenizer, method, num_classes):
#    tokens, label_ids = map(list, zip(*batch))
#    text_ids = tokenizer(tokens,
#                         padding=True,
#                         truncation=True,
#                         max_length=256,
#                         is_split_into_words=True,
#                         add_special_tokens=True,
#                         return_tensors='pt')
#    if method not in ['ce', 'scl']:
#        positions = torch.zeros_like(text_ids['input_ids'])
#        positions[:, num_classes:] = torch.arange(0, text_ids['input_ids'].size(1)-num_classes)
#        text_ids['position_ids'] = positions
#    return text_ids, torch.tensor(label_ids)


def load_data(dataset, data_dir, tokenizer, test_batch_size, model_name, method, workers):
    if dataset == 'sst2':
        train_data = json.load(open(os.path.join(data_dir, 'SST2_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SST2_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'trec':
        train_data = json.load(open(os.path.join(data_dir, 'TREC_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'TREC_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'description': 0, 'entity': 1, 'abbreviation': 2, 'human': 3, 'location': 4, 'numeric': 5}
    elif dataset == 'cr':
        train_data = json.load(open(os.path.join(data_dir, 'CR_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'CR_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'subj':
        train_data = json.load(open(os.path.join(data_dir, 'SUBJ_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SUBJ_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'subjective': 0, 'objective': 1}
    elif dataset == 'pc':
        train_data = json.load(open(os.path.join(data_dir, 'procon_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'procon_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'oads':
        # train_data = json.load(open(os.path.join(data_dir, 'OADS_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'OADS_Test.json'), 'r', encoding='utf-8'))
        print(os.path.join(data_dir, 'OADS_Test.json'))
        # test_data = json.load(open(os.path.join(data_dir, 'predicted-output.json'), 'r', encoding='utf-8'))
        # label_dict2 = {'general-url': 0, 'third-party-dataset': 1, 'author-provided-dataset': 2, 'third-party-software': 3, 'author-provided-software': 4}
        label_dict = {'general-url': 0, 'third-party-dataset': 1, 'author-provided-dataset': 2, 'third-party-software': 3, 'author-provided-software': 4, 'project': 5}
    else:
        raise ValueError('unknown dataset')
    # trainset = MyDataset(train_data, label_dict, tokenizer, model_name, method)
    testset = MyDataset(test_data, label_dict, tokenizer, model_name, method)
        #collate_fn = partial(my_collate, tokenizer=tokenizer, method=method, num_classes=len(label_dict))
    # train_dataloader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn, pin_memory=True)
        #test_dataloader = DataLoader(testset, test_batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn, pin_memory=True)
    # return train_dataloader, test_dataloader
    # Use the simple my_collate that handles pre-tokenized dicts
    test_dataloader = DataLoader(
        testset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=my_collate,  # No partial, no extra args
        pin_memory=True
    )
    return test_dataloader

