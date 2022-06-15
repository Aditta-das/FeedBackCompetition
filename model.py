import torch
import torch.nn as nn
import transformers
from config import Config

class FeedBackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(Config.MODEL_NAME, return_dict=False)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, Config.CLASSES)
    
    def forward(self, ids, mask, token_type_ids):
        x = self.bert(ids, mask, token_type_ids)
        x = self.dropout(x)
        x = self.fc(x)
        return x