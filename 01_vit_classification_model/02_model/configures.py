import torch

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':1  ,
    'LEARNING_RATE':2e-5,
    'BATCH_SIZE':8,
    'SEED':42,
    'DEVICE': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'MODEL_DIR': r'../runs',
    'PROJECT':'condition_classification'
}