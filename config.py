import torch
import transformers
from transformers import AutoTokenizer, BertTokenizer

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LR = [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7]
    TRAIN_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 8
    EPOCHS = 10
    N_FOLD = 5
    TOKENIZER = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
    CLASSES = 3
    MAX_LEN = 100
    TRAIN_CSV = "../train.csv"
    TEST_CSV = "../test.csv"
    API = "9d02c7851f62695a82c1e14023cd456fc37d629b"
    PROJECT_NAME = "feedback-wandb"
    MODEL_NAME = "bert-large-uncased"