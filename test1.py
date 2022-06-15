import argparse
args = argparse.ArgumentParser()
args.add_argument("--fold", type=int)
args.add_argument("--epochs", type=int)
args.add_argument("--path", type=str)
cargs = args.parse_args()

fold = cargs.fold
epoch = cargs.epochs
path = cargs.path
print(f"print fold: {fold} and epoch: {epoch} path: {path}")
# import os
# import pandas as pd
# from sklearn import model_selection
# def fetchEssay(essay_id):
#     """
#     Read the text file of the specific essay_id
#     """
#     essay_path = os.path.join("/media/aditta/UBUNTU/feedback-prize-effectiveness/train", essay_id + '.txt')
#     essay_text = open(essay_path, 'r').read()
#     return essay_text

# train = pd.read_csv("../train.csv")
# n = train.copy()
# n["essay"] = n['essay_id'].apply(fetchEssay)
# # print(n)
# print("##################")
# # print(n['essay'][1])

# n = n.drop(['discourse_text', 'essay'], axis=1)
# print(n)