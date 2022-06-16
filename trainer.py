import argparse
import torch
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from config import Config
from wandbViz import wandb_log, wand
from dataset import FeedBack
from sklearn import model_selection
from model import FeedBackModel
import torch.nn as nn

args = argparse.ArgumentParser()
args.add_argument("--fold", type=int)
args.add_argument("--epochs", type=int)
args.add_argument("--path", type=str)
args.add_argument("--name", type=str)
cargs = args.parse_args()

class FeedBackTrainer:
    def __init__(self, dataloader, optimizer, scheduler, model, loss_fn, device=Config.DEVICE):
        self.train_loader, self.test_loader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.train_loss, self.test_loss = loss_fn
        self.device = device

    def train_one_epoch(self):
        self.model.train()
        loop = tqdm(self.train_loader, total=len(self.train_loader))
        train_preds, train_labels = [], []
        running_loss = 0
        for batch_idx, data in enumerate(loop):
            ids = data["ids"].to(self.device)
            mask = data["mask"].to(self.device)
            token_type_ids = data["token_type_ids"].to(self.device)
            target = data["targets"].to(self.device)
            self.optimizer.zero_grad()
            output = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = self.train_loss(output, target)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            loss_train = loss.item()
            running_loss += loss_train

            loop.set_description(desc=f"loss: {loss_train}")

            train_preds += [output.detach().cpu().numpy()]
            train_labels += [target.detach().cpu().numpy()]
        all_train_preds = np.concatenate(train_preds)
        all_train_labels = np.concatenate(train_labels)

        return running_loss, running_loss / len(self.train_loader)


    def test_one_epoch(self):
        self.model.eval()
        running_val_loss = 0
        valid_preds, valid_targets = [], []
        loop = tqdm(self.test_loader, total=len(self.train_loader))
        with torch.no_grad():
            for batch_idx, data in enumerate(loop):
                ids = data["ids"].to(self.device)
                mask = data["mask"].to(self.device)
                token_type_ids = data["token_type_ids"].to(self.device)
                target = data["targets"].to(self.device)
                output = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                test_loss = self.test_loss(output, target)
                loss_test = test_loss.item()
                running_val_loss += loss_test

                loop.set_description(desc=f"loss: {loss_test}")

                valid_preds += [output.detach().cpu().numpy()]
                valid_targets += [target.detach().cpu().numpy()]
            all_test_preds = np.concatenate(valid_preds)
            all_test_labels = np.concatenate(valid_targets)

            return all_test_preds, all_test_labels, running_val_loss, running_val_loss / len(self.test_loader)

    def fit(self):
        best_loss = np.inf
        for epoch in range(cargs.epochs):
            print(f"[INFO] EPOCHS: {epoch+1} / {cargs.epochs}")
            running_loss, train_running_loss = self.train_one_epoch()
            '''
            print: train loss
            '''
            test_preds, test_targets, running_val_loss, test_loss = self.test_one_epoch()
            '''
            print: test loss
            '''
            if test_loss < best_loss:
                best_loss = test_loss
                '''
                self.model_save() function call
                print best_loss
                '''
                self.model_saving(cargs.path, cargs.name)
                print(f"[INFO] best loss: {best_loss}")

            # Wandb log
            wandb_log(
                running_tloss=running_loss,
                running_vloss=running_val_loss,
                train_loss=train_running_loss,
                test_loss=test_loss
            )

    def model_saving(self):
        try:
            if not os.path.exists(cargs.path):
                os.makedirs(cargs.path)
        except:
            print("[ERROR] error while making directory")

        save_path = os.path.join(cargs.path, cargs.name)
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(checkpoint, save_path)


if __name__ == "__main__":
    train = pd.read_csv(Config.TRAIN_CSV)
    run = wand()
    '''
    process of data make kfold, dataloader, and define everything
    '''
    train["discourse_effectiveness"] = train["discourse_effectiveness"].map(
        {'Ineffective': 2, 'Adequate': 0, 'Effective': 1}
    )
    
    kf = model_selection.StratifiedKFold(n_splits=Config.N_FOLD)
    for fold_, (train_idx, test_idx) in enumerate(kf.split(X=train, y=train["discourse_type"])):
        print(f"[INFO] Fold: {fold_+1}/{Config.N_FOLD}")
        df_train = train.loc[train_idx]
        df_test = train.loc[test_idx]

        train_dataset = FeedBack(
            df_train,
            is_test=False
        )
        test_dataset = FeedBack(
            df_test,
            is_test=False
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )


        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.TEST_BATCH_SIZE,
            shuffle=False
        )

        model = FeedBackModel()
        model.to(Config.DEVICE)
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
            },
            {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        ]
        
        run.watch(model)
        
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=Config.LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=20, 
            eta_min=1e-4
        )
        train_fn = nn.CrossEntropyLoss()
        test_fn = nn.CrossEntropyLoss()

        trainer = FeedBackTrainer(
            dataloader=(train_loader, test_loader),
            loss_fn=(train_fn, test_fn),
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
        )

        trainer.fit()

    run.finish()