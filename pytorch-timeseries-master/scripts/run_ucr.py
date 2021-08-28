"""
Example scripts demonstrating how the UCRTrainer, which extends the BaseTrainer,
can be used to train an Inception Model on UCR Archive data.

The ECG200 dataset has 1 output class, while the Synthetic Control dataset has
6 - in the case of ECG, a sigmoid function is used as the final activation function.
For the Synthetic Control dataset, softmax is used instead.
"""
from pathlib import Path
import sys

sys.path.append('..')

from src import UCRTrainer, load_ucr_trainer, HARTrainer, load_har_trainer, OPPTrainer, load_opp_trainer, PAMAPTrainer, load_pamap_trainer, \
    MHEALTHTrainer, load_mhealth_trainer, WISDMTrainer, load_wisdm_trainer, wHARTrainer, load_whar_trainer
from src.models import InceptionModel, LinearBaseline, FCNBaseline, ResNetBaseline

# def train_inception_ecg():
#     data_folder = Path('../data')
#
#     model = InceptionModel(num_blocks=1, in_channels=1, out_channels=2,
#                            bottleneck_channels=2, kernel_sizes=41, use_residuals=True,
#                            num_pred_classes=1)
#
#     trainer = UCRTrainer(model=model, experiment='ECG200', data_folder=data_folder)
#     trainer.fit()
#
#     savepath = trainer.save_model()
#     new_trainer = load_ucr_trainer(savepath)
#     new_trainer.evaluate()

def train_inception_har():
    data_folder = Path('../data/UCI_HAR_Dataset')
    model = InceptionModel(num_blocks=1, in_channels=9, out_channels=32,
                           bottleneck_channels=2, kernel_sizes=20, use_residuals=True,
                           num_pred_classes=6)

    trainer = HARTrainer(model=model, data_folder=data_folder)
    trainer.fit()

    savepath = trainer.save_model()
    new_trainer = load_har_trainer(savepath)
    new_trainer.evaluate()

def train_inception_opp():
    model = InceptionModel(num_blocks=1, in_channels=113, out_channels=32,
                           bottleneck_channels=2, kernel_sizes=20, use_residuals=True,
                           num_pred_classes=18)

    trainer = OPPTrainer(model=model)
    trainer.fit()

    savepath = trainer.save_model()
    new_trainer = load_opp_trainer(savepath)
    new_trainer.evaluate()

def train_inception_pamap():
    model = InceptionModel(num_blocks=1, in_channels=40, out_channels=32,
                           bottleneck_channels=2, kernel_sizes=20, use_residuals=True,
                           num_pred_classes=18)

    trainer = PAMAPTrainer(model=model)
    trainer.fit()

    savepath = trainer.save_model()
    #savepath = Path("data/models/InceptionModel/InceptionModel_model_pamap_20_epochs.pkl")
    new_trainer = load_pamap_trainer(savepath)
    new_trainer.evaluate()

def train_inception_mhealth():
    model = InceptionModel(num_blocks=1, in_channels=23, out_channels=32,
                           bottleneck_channels=2, kernel_sizes=20, use_residuals=True,
                           num_pred_classes=12)

    trainer = MHEALTHTrainer(model=model)
    trainer.fit()

    savepath = trainer.save_model()
    #savepath = Path("data/models/InceptionModel/InceptionModel_model_pamap_20_epochs.pkl")
    new_trainer = load_mhealth_trainer(savepath)
    new_trainer.evaluate()

def train_inception_wisdm():
    model = InceptionModel(num_blocks=1, in_channels=3, out_channels=32,
                           bottleneck_channels=2, kernel_sizes=20, use_residuals=True,
                           num_pred_classes=6)

    trainer = WISDMTrainer(model=model)
    trainer.fit()

    savepath = trainer.save_model()
    #savepath = Path("data/models/InceptionModel/InceptionModel_model_pamap_20_epochs.pkl")
    new_trainer = load_wisdm_trainer(savepath)
    new_trainer.evaluate()

def train_inception_whar():
    model = InceptionModel(num_blocks=1, in_channels=120, out_channels=32,
                           bottleneck_channels=2, kernel_sizes=20, use_residuals=True,
                           num_pred_classes=8)

    trainer = wHARTrainer(model=model)
    trainer.fit()

    savepath = trainer.save_model()
    #savepath = Path("data/models/InceptionModel/InceptionModel_model_pamap_20_epochs.pkl")
    new_trainer = load_whar_trainer(savepath)
    new_trainer.evaluate()

# def train_linear_ecg():
#     data_folder = Path('../data')
#
#     model = LinearBaseline(num_inputs=96, num_pred_classes=1)
#
#     trainer = UCRTrainer(model=model, experiment='ECG200', data_folder=data_folder)
#     trainer.fit()
#
#     savepath = trainer.save_model()
#     new_trainer = load_ucr_trainer(savepath)
#     new_trainer.evaluate()


# def train_fcn_ecg():
# #     data_folder = Path('../data')
# #
# #     model = FCNBaseline(in_channels=1, num_pred_classes=1)
# #
# #     trainer = UCRTrainer(model=model, experiment='ECG200', data_folder=data_folder)
# #     trainer.fit()
# #
# #     savepath = trainer.save_model()
# #     new_trainer = load_ucr_trainer(savepath)
# #     new_trainer.evaluate()
# #
# #
# # def train_resnet_ecg():
# #     data_folder = Path('../data')
# #
# #     model = ResNetBaseline(in_channels=1, num_pred_classes=1)
# #
# #     trainer = UCRTrainer(model=model, experiment='ECG200', data_folder=data_folder)
# #     trainer.fit()
# #
# #     savepath = trainer.save_model()
# #     new_trainer = load_ucr_trainer(savepath)
# #     new_trainer.evaluate()
# #
# #
# # def train_inception_sc():
# #     data_folder = Path('../data')
# #
# #     model = InceptionModel(num_blocks=1, in_channels=1, out_channels=2,
# #                            bottleneck_channels=2, kernel_sizes=41, use_residuals=True,
# #                            num_pred_classes=6)
# #
# #     trainer = UCRTrainer(model=model, experiment='synthetic_control', data_folder=data_folder)
# #     trainer.fit()
# #
# #     savepath = trainer.save_model()
# #     new_trainer = load_ucr_trainer(savepath)
# #     new_trainer.evaluate()


if __name__ == '__main__':
    #train_inception_opp()
    #train_inception_har()
    train_inception_pamap()
    #train_inception_mhealth()
    #train_inception_wisdm()
    #train_inception_whar()
    # train_linear_ecg()
    # train_fcn_ecg()
    # train_resnet_ecg()
