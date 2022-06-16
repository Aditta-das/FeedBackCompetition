import torch
from config import Config
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd

class FeedBack(Dataset):
    def __init__(self, data, is_test=False):
        self.data = data
        self.config = Config()
        self.is_test = is_test
    
    def __getitem__(self, idx):
        text = self.data['discourse_text'].iloc[idx]
        if not self.is_test:
            target_value = self.data['discourse_effectiveness'].iloc[idx]
                
        inputs = self.config.TOKENIZER.encode_plus(
            text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.config.MAX_LEN,
            padding='max_length'
        )
        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)
        
        if self.is_test:
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
            }
        else:
            targets = torch.tensor(target_value, dtype=torch.long)
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
                'targets': targets
            }
        
    def __len__(self):
        return len(self.data)

# train = pd.read_csv(Config.TRAIN_CSV)
# train['discourse_effectiveness'] = train['discourse_effectiveness'].map(
#         {'Ineffective': 0, 'Adequate': 1, 'Effective': 2}
#     )
# for i in range(5):
#     print(BERTDataset(train).__getitem__(i)["ids"].shape)