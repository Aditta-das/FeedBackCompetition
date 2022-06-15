import torch
from config import Config
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd

class FeedBack(Dataset):
    def __init__(self, data, is_test=False):
        super(FeedBack, self).__init__()
        self.data = data
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data["discourse_text"].iloc[idx]
        if not self.is_test:
            label = self.data["discourse_effectiveness"].iloc[idx]
        inputs = Config.TOKENIZER.encode_plus(
            data,
            None,
            truncation=True,
            max_length=Config.MAX_LEN,
            padding=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        if self.is_test:
            return {
                "ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            }
        else:
            return {
                "ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "target": torch.tensor(label, dtype=torch.long) 
            }

# train = pd.read_csv(Config.TRAIN_CSV)
# train['discourse_effectiveness'] = train['discourse_effectiveness'].map(
#         {'Ineffective': 0, 'Adequate': 1, 'Effective': 2}
#     )
# print(FeedBack(train).__getitem__(10))