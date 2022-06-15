import wandb
from config import Config

def wand():
    wandb.login(key=Config.API)
    run = wandb.init(
        project=Config.PROJECT_NAME
    )
    return run

def wandb_log(**kwargs):
    for k, v in kwargs.items():
        wandb.log({k:v})