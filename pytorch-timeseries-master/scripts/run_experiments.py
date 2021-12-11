"""
Example scripts demonstrating how the UCRTrainer, which extends the BaseTrainer,
can be used to train an Inception Model on UCR Archive data.

The ECG200 dataset has 1 output class, while the Synthetic Control dataset has
6 - in the case of ECG, a sigmoid function is used as the final activation function.
For the Synthetic Control dataset, softmax is used instead.
"""
from pathlib import Path
import sys
import wandb
import torch.cuda
import torch
import argparse
import math
sys.path.append('..')

from src import UCRTrainer, load_ucr_trainer, HARTrainer, load_har_trainer, OPPTrainer, load_opp_trainer, PAMAPTrainer, load_pamap_trainer, \
    MHEALTHTrainer, load_mhealth_trainer, WISDMTrainer, load_wisdm_trainer, wHARTrainer, load_whar_trainer
from src.models import InceptionModel

sweep_config = {
                'method': 'grid',
                'metric': {'goal': 'maximize', 'name': 'acc'},
                'parameters': {
                    'batch_size': {'values': [32, 64, 128, 256]},
                    'lr_scheduler': {'values': ['MultiStepLR', 'StepLR', 'ExponentialLR']},
                    'epochs': {'values': [20, 50 ,100]},
                    'weight_decay': {'values': [0.9, 0.99, 0.5]},
                    #'fc_layer_size': {'values': [128, 256, 512]},
                    'learning_rate': {'values': [0.01, 0.001]},
                    'optimizer': {'values': ['adam', 'sgd', 'adamw']}
                }
}

sweep_id = wandb.sweep(sweep_config, project="HAR", entity="conset_adl")

def train_inception_har():
    print("Running HAR")
    data_folder = Path('../data/UCI_HAR_Dataset')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = InceptionModel(num_blocks=1, in_channels=9, out_channels=32,
                           bottleneck_channels=2, kernel_sizes=20, use_residuals=True,
                           num_pred_classes=6)
    model.to(device)
    trainer = HARTrainer(model=model, data_folder=data_folder)
    #trainer.fit()

    wandb.agent(sweep_id, function=trainer.fit, count=648)

    #savepath = trainer.save_model()
    #new_trainer = load_har_trainer(savepath)
    #new_trainer.evaluate()

def train_inception_opp():
    print("Running Opportunity")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InceptionModel(num_blocks=1, in_channels=113, out_channels=32,
                           bottleneck_channels=2, kernel_sizes=20, use_residuals=True,
                           num_pred_classes=17)
    model.to(device)
    trainer = OPPTrainer(model=model)
    #trainer.fit()

    wandb.agent(sweep_id, function=trainer.fit, count=15)

    #savepath = trainer.save_model()
    #savepath = Path("data/models/InceptionModel/InceptionModel_model_opportunity_20_epochs.pkl")
    #new_trainer = load_opp_trainer(savepath)
    #new_trainer.evaluate()

def train_inception_pamap():
    print("Running PAMAP2")
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = InceptionModel(num_blocks=1, in_channels=1, out_channels=32,
                           bottleneck_channels=2, kernel_sizes=20, use_residuals=True,
                           num_pred_classes=18)
    model.to(device)
    trainer = PAMAPTrainer(model=model)
    #trainer.fit()

    wandb.agent(sweep_id, function=trainer.fit, count=15)

    #savepath = trainer.save_model()
    #savepath = Path("data/models/InceptionModel/InceptionModel_model_pamap_20_epochs.pkl")
    #new_trainer = load_pamap_trainer(savepath)
    #new_trainer.evaluate()

def train_inception_mhealth():
    print("Running MHEALTH")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InceptionModel(num_blocks=1, in_channels=23, out_channels=32,
                           bottleneck_channels=2, kernel_sizes=20, use_residuals=True,
                           num_pred_classes=12)
    model.to(device)
    trainer = MHEALTHTrainer(model=model)
    #trainer.fit()

    wandb.agent(sweep_id, function=trainer.fit, count=15)

    #savepath = trainer.save_model()
    #savepath = Path("data/models/InceptionModel/InceptionModel_model_mhealth_20_epochs.pkl")
    #new_trainer = load_mhealth_trainer(savepath)
    #new_trainer.evaluate()

def train_inception_wisdm():
    print("Running WISDM")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InceptionModel(num_blocks=1, in_channels=3, out_channels=32,
                           bottleneck_channels=2, kernel_sizes=20, use_residuals=True,
                           num_pred_classes=6)
    model.to(device)
    trainer = WISDMTrainer(model=model)
    trainer.fit()

    savepath = trainer.save_model()
    #savepath = Path("data/models/InceptionModel/InceptionModel_model_wisdm_20_epochs.pkl")
    new_trainer = load_wisdm_trainer(savepath)
    new_trainer.evaluate()

def train_inception_whar():
    print("Running WHAR")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InceptionModel(num_blocks=1, in_channels=120, out_channels=32,
                           bottleneck_channels=2, kernel_sizes=20, use_residuals=True,
                           num_pred_classes=8)
    model.to(device)
    trainer = wHARTrainer(model=model)
    #trainer.fit()

    wandb.agent(sweep_id, function=trainer.fit, count=15)

    #savepath = trainer.save_model()
    #savepath = Path("data/models/InceptionModel/InceptionModel_model_pamap_20_epochs.pkl")
    #new_trainer = load_whar_trainer(savepath)
    #new_trainer.evaluate()

if __name__ == '__main__':
    #train_inception_opp()
    train_inception_har()
    #train_inception_pamap()
    #train_inception_mhealth()
    #train_inception_wisdm()
    #train_inception_whar()
    # train_linear_ecg()
    # train_fcn_ecg()
    # train_resnet_ecg()
