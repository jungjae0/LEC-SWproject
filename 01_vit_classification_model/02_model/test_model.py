import pandas as pd
import os
import re
import tqdm
from sklearn.metrics import accuracy_score

from models import CNNModel, VITModel
from dataset import make_loader
from configures import CFG

import torch
import torch.nn as nn
import torch.optim as optim

from multiprocessing import Manager, freeze_support

def eval_fn(data_loader, model, criterion, epoch_loss = 0.0):
    model.eval().to(CFG['DEVICE'])
    criterion.eval()

    preds = []
    actuals = []

    with torch.no_grad():
        for i_batch, item in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
            images = item['image'].to(CFG['DEVICE'])
            labels = item['label'].to(CFG['DEVICE'])

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            val_loss = criterion(outputs, labels)

            epoch_loss += val_loss.item()

            preds.extend(predicted.tolist())  # Add predicted values to the list
            actuals.extend(labels.tolist())  # Add actual values to the list

    return epoch_loss, preds, actuals


def run(model_path):

    criterion = nn.CrossEntropyLoss()

    model = VITModel(num_classes)
    model.load_state_dict(torch.load(model_path))

    # valid
    val_epoch_loss, val_preds, val_actuals = eval_fn(val_loader, model, criterion)


    val_acc = accuracy_score(val_actuals, val_preds)
    val_loss = val_epoch_loss / len(val_loader)
    print(val_preds)

    return val_acc, val_loss

img_dir = r"C:\code\LEC-SWproject\02_leaf_segmentation\classification_imgs"
files = os.listdir(img_dir)
df = pd.DataFrame(files, columns=['image'])
df['label'] = 0
df['img_path'] = img_dir + "\\" + df['image']


manager = Manager()
img_cache = manager.dict()

# val = pd.read_csv("../../02_classification/test_dataset.csv")
val = df

num_classes = 3
val_loader = make_loader(val, batch_size=CFG['BATCH_SIZE'], cache=img_cache, data_type='valid')

model_path = r"D:\sw_models\vit-learn\vit-learn_99.pth"

val_acc, val_loss = run(model_path)
print(val_acc, val_loss)