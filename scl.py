import torch
import pandas as pd
from tqdm import tqdm
from model import Transformer
from config import get_config
from loss_func import CELoss, SupConLoss, DualLoss
from data_utils import load_data
from transformers import logging, AutoTokenizer, AutoModel
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.metrics import classification_report
import pickle
import numpy as np

class Instructor:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))

        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            base_model = AutoModel.from_pretrained('bert-base-uncased')
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
            base_model = AutoModel.from_pretrained('roberta-base')
        elif args.model_name == 'DistilBERT':
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            base_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        elif args.model_name == 'GPT-2':
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT2 needs pad token
            base_model = GPT2Model.from_pretrained("gpt2")
        else:
            raise ValueError('unknown model')

        self.model = Transformer(base_model, args.num_classes, args.method)
        self.model.to(args.device)

        # Load pretrained weights
        best_model_file_path = 'saved_weights/SCL_weights.pkl'
        with open(best_model_file_path, 'rb') as model_file:
            best_model_state_dict = pickle.load(model_file)
        self.model.load_state_dict(best_model_state_dict, strict=False)

        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _test(self, dataloader, criterion):
        self.model.eval()
        all_predictions = []
        all_labels = []          # Will stay empty if no labels
        total_loss = 0.0
        n_samples = 0
        has_labels = False

        with torch.no_grad():
            for batch_inputs, targets in tqdm(dataloader, disable=self.args.backend):
                # Move inputs to device
                input_ids = batch_inputs['input_ids'].to(self.args.device)
                attention_mask = batch_inputs['attention_mask'].to(self.args.device)
                model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

                # Forward pass
                outputs = self.model(model_inputs)

                # Assume outputs is logits tensor (common for classification head)
                # If your Transformer returns a dict with 'predicts', change to: logits = outputs['predicts']
                #logits = outputs  # <-- MOST LIKELY CORRECT FOR YOUR SETUP
                logits = outputs['predicts']
                preds = torch.argmax(logits, dim=-1)
                all_predictions.extend(preds.cpu().numpy())

                # Handle case where targets are None (inference only)
                if targets is not None:
                    has_labels = True
                    targets = targets.to(self.args.device)

                    # Only compute loss if the method actually supports it in test mode
                    # Contrastive losses (scl, dualcl) require cls_feats which are not available in eval
                    if self.args.method == 'ce':
                        loss = criterion(logits, targets)
                        total_loss += loss.item() * targets.size(0)
                    # For scl / dualcl: skip loss computation (it's not meaningful without augmentations)
                    else:
                        pass  # loss remains 0 or None

                        n_samples += targets.size(0)
                        all_labels.extend(targets.cpu().numpy())
                else:
                    n_samples += input_ids.size(0)  # Count samples even without labels

        # Compute metrics only if labels exist
        if has_labels and n_samples > 0:
            avg_loss = total_loss / n_samples
            accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        else:
            avg_loss = None
            accuracy = None

        return avg_loss, accuracy, np.array(all_labels) if all_labels else None, np.array(all_predictions)

    def run(self):
        test_dataloader = load_data(
            dataset=self.args.dataset,
            data_dir=self.args.data_dir,
            tokenizer=self.tokenizer,
            test_batch_size=self.args.test_batch_size,
            model_name=self.args.model_name,
            method=self.args.method,
            workers=0
        )

        # Criterion is only needed if labels exist — but we create it anyway
        if self.args.method == 'ce':
            criterion = CELoss()
        elif self.args.method == 'scl':
            criterion = SupConLoss(self.args.alpha, self.args.temp)
        elif self.args.method == 'dualcl':
            criterion = DualLoss(self.args.alpha, self.args.temp)
        else:
            raise ValueError('unknown method')

        test_loss, test_acc, true_labels, predictions = self._test(test_dataloader, criterion)

        if test_acc is not None:
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}%'.format(test_loss, test_acc * 100))
            print(classification_report(true_labels, predictions))
        else:
            self.logger.info('[inference mode] No ground truth labels — generating predictions only.')
            print(f"Generated {len(predictions)} predictions.")

        print("Sample predictions:", predictions[:10])

        return true_labels, predictions


if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    ins = Instructor(args, logger)
    true_labels, scl_preds = ins.run()