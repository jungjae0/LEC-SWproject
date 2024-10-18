import os
import tqdm
import cv2
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from models import CNNModel, VITModel
from dataset import make_loader
from configures import CFG

from multiprocessing import Manager


def train_fn(data_loader, model, criterion, epoch_loss = 0.0):
    model.train()

    preds = []
    actuals = []

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])

    for i_batch, item in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        # print(item['label'])
        images = item['image'].to(CFG['DEVICE'], non_blocking=True)
        labels = item['label'].to(CFG['DEVICE'], non_blocking=True)

        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds.extend(predicted.tolist())
        actuals.extend(labels.tolist())

        epoch_loss += loss.item()

    return epoch_loss, preds, actuals

def eval_fn(data_loader, model, criterion, epoch_loss = 0.0):
    model.eval()
    criterion.eval()

    preds = []
    actuals = []

    with torch.no_grad():
        for i_batch, item in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
            images = item['image'].to(CFG['DEVICE'], non_blocking=True)
            labels = item['label'].to(CFG['DEVICE'], non_blocking=True)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            val_loss = criterion(outputs, labels)

            epoch_loss += val_loss.item()

            preds.extend(predicted.tolist())  # Add predicted values to the list
            actuals.extend(labels.tolist())  # Add actual values to the list

    return epoch_loss, preds, actuals


def run(model, train_loader, val_loader, model_name):
    wandb.init(project=CFG['PROJECT'], entity='jyihaan4', name=f'{model_name}')

    wandb.config = {
        "learning_rate": CFG['LEARNING_RATE'],
        "epochs": CFG['EPOCHS'],
        "batch_size": CFG['BATCH_SIZE']
    }


    model_dir = os.path.join(CFG['MODEL_DIR'], f"{model_name}")
    os.makedirs(model_dir, exist_ok=True)


    model = model.to(CFG['DEVICE'])

    criterion = nn.CrossEntropyLoss()

    epoch_list = []
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []

    for epoch in range(CFG['EPOCHS']):
        # train
        train_epoch_loss, train_preds, train_actuals = train_fn(train_loader, model, criterion)

        # valid
        val_epoch_loss, val_preds, val_actuals = eval_fn(val_loader, model, criterion)

        # acc, loss
        train_acc = accuracy_score(train_actuals, train_preds)
        train_loss = train_epoch_loss / len(train_loader)

        val_acc = accuracy_score(val_actuals, val_preds)
        val_loss = val_epoch_loss / len(val_loader)

        epoch_list.append(epoch)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        print(f'Epoch [{epoch + 1}/{CFG["EPOCHS"]}], '
              f'Train Loss: {train_loss}, '
              f'Train Accuracy: {train_acc}, '
              f'Val Loss: {val_loss}, '
              f'Val Accuracy: {val_acc}')

        wandb.log({"Train Accuracy": train_acc,
                   "Train Loss": train_loss,
                   "Val Accuracy": val_acc,
                   "Val Loss": val_loss})

        torch.save(model.state_dict(), os.path.join(model_dir, f'{model_name}_{epoch + 1}.pth'))

    data = {
        'epoch': epoch_list,
        'train_acc': train_acc_list,
        'train_loss': train_loss_list,
        'val_acc': val_acc_list,
        'val_loss': val_loss_list
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(model_dir, f'results_{model_name}_cache.csv'), index=False)
def run_test(model, val_loader):
    model.to(CFG['DEVICE'])

    criterion = nn.CrossEntropyLoss()

    # valid
    val_epoch_loss, val_preds, val_actuals = eval_fn(val_loader, model, criterion)


    val_acc = accuracy_score(val_actuals, val_preds)
    val_loss = val_epoch_loss / len(val_loader)

    print(val_acc, val_loss)


    return val_acc, val_loss
def main():

    manager = Manager()
    img_cache = manager.dict()
    train = pd.read_csv("../../02_classification/train_dataset.csv")
    valid = pd.read_csv("../../02_classification/val_dataset.csv")
    # train_loader = make_loader(train, batch_size=CFG['BATCH_SIZE'], cache=img_cache, data_type='train')
    # val_loader = make_loader(valid, batch_size=CFG['BATCH_SIZE'], cache=img_cache, data_type='valid')

    model_path = r"D:\sw_models\vit-learn\vit-learn_99.pth"

    num_classes = 3

    # cnn_model = CNNModel(num_classes)
    vit_model = VITModel(num_classes)

    print("모델 준비 완료")
    # train
    # cnn
    # run(cnn_model, train_loader, val_loader, model_name='cnn')
    # vit
    print("모델 학습 시작")
    vit_model.load_state_dict(torch.load(model_path))
    test = pd.read_csv("../../02_classification/test_dataset.csv")
    test_loader = make_loader(test, batch_size=CFG['BATCH_SIZE'], cache=img_cache, data_type='valid')
    run_test(vit_model, test_loader)
    # run(vit_model, train_loader, val_loader, model_name='vit-learn')


if __name__ == '__main__':
    main()