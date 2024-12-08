from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import scikitplot as skplt
import torch
from torch import nn, optim
import seaborn as sns
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from .labels import *
import matplotlib
import wandb
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, StepLR
import math

from typing import cast, Any, Dict, List, Tuple, Optional

CURRENT_DATASET = 'w-HAR'

HAR_TRAIN_SIZE = 7352
HAR_VAL_SIZE = 736

wHAR_TRAIN_SIZE = 2841
wHAR_VAL_SIZE = 944

MHEALTH_TRAIN_SIZE = 247093
MHEALTH_VAL_SIZE = 27455

PAMAP_TRAIN_SIZE = 426102
PAMAP_VAL_SIZE = 47345

OPP_TRAIN_SIZE = 502166
OPP_VAL_SIZE = 55797

WISDM_TRAIN_SIZE = 790576
WISDM_VAL_SIZE = 87842

SMALL_SIZE = 10
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#La fiecare rulare de experiment diferita, trebuie modificat:
# - TRAIN SIZE SI VAL SIZE
# - Variabile pentru wandb
# - Experimentul rulat in run_experiments
class BaseTrainer:

    """Trains an inception model. Dataset-specific trainers should extend this class
    and implement __init__, get_loaders and save functions.
    See UCRTrainer in .ucr.py for an example.

    Attributes
    ----------
    The following need to be added by the initializer:
    model:
        The initialized inception model
    data_folder:
        A path to the data folder - get_loaders should look here for the data
    model_dir:
        A path to where the model and its predictions should be saved

    The following don't:
    train_loss:
        The fit function fills this list in as the model trains. Useful for plotting
    val_loss:
        The fit function fills this list in as the model trains. Useful for plotting
    test_results:
        The evaluate function fills this in, evaluating the model on the test data
    """

    model: nn.Module
    data_folder: Path
    model_dir: Path
    train_loss: List[float] = []
    val_loss: List[float] = []
    test_results: Dict[str, float] = {}
    input_args: Dict[str, Any] = {}
    history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

    def fit(self, batch_size: int = 64, num_epochs: int = 20,
            val_size: float = 0.2, learning_rate: float = 0.01,
            patience: int = 40, config = None) -> None:
        """Trains the inception model

        Arguments
        ----------
        batch_size:
            Batch size to use for training and validation
        num_epochs:
            Maximum number of epochs to train for
        val_size:
            Fraction of training set to use for validation
        learning_rate:
            Learning rate to use with Adam optimizer
        patience:
            Maximum number of epochs to wait without improvement before
            early stopping
        """
        with wandb.init(project="MHEALTH", entity="conset_adl", config=config):
            config = wandb.config
            train_loader, val_loader = self.get_loaders(config.batch_size, mode='train', val_size=val_size)

            if config.optimizer == "sgd":
                optimizer = optim.SGD(self.model.parameters(),
                                      lr=0.01, momentum=0.9, weight_decay=config.weight_decay)
                if config.lr_scheduler == "MultiStepLR":
                    MultiStepLR(optimizer, milestones=[15, 35], gamma=0.1)
                elif config.lr_scheduler == "StepLR":
                    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
                elif config.lr_scheduler == "ExponentialLR":
                    scheduler = ExponentialLR(optimizer, gamma=0.9)

            elif config.optimizer == "adam":
                optimizer = optim.Adam(self.model.parameters(),
                                       lr=config.learning_rate, weight_decay=config.weight_decay)
            elif config.optimizer == "adamw":
                optimizer = optim.AdamW(self.model.parameters(),
                                       lr=config.learning_rate, weight_decay=config.weight_decay)

            best_val_loss = np.inf
            patience_counter = 0
            best_state_dict = None
            
            # model re-init by loading default initial state_dict
            
            self.model.train()
            print('Learning rate: ', config.learning_rate)
            print('Optimizer: ', config.optimizer)
            print('Batch size: ', config.batch_size)
            print('Num of epochs: ', config.epochs)
            for epoch in range(config.epochs):
                val_corrects = 0.0
                train_corrects = 0.0
                epoch_train_loss = []
                for x_t, y_t in train_loader:
                    #x_t, y_t = x_t.cuda(), y_t.cuda()
                    optimizer.zero_grad()
                    output = self.model(x_t)
                    _, preds = torch.max(output.data, 1)

                    if len(y_t.shape) == 1:
                        train_loss = F.binary_cross_entropy_with_logits(
                            output, y_t.unsqueeze(-1).float(), reduction='mean'
                        )
                    else:
                        train_loss = F.cross_entropy(output, y_t.argmax(dim=-1), reduction='mean')
                    _, correct_label = torch.max(y_t.data, 1)
                    iter_corrects = torch.sum(preds == correct_label).to(torch.float32)
                    train_corrects += iter_corrects

                    epoch_train_loss.append(train_loss.item())
                    train_loss.backward()
                    optimizer.step()

                epoch_acc_train = train_corrects / HAR_TRAIN_SIZE

                #print('Train accuracy: ', epoch_acc_train)

                self.train_loss.append(np.mean(epoch_train_loss))

                epoch_val_loss = []
                self.model.eval()
                for x_v, y_v in cast(DataLoader, val_loader):
                    #x_v, y_v = x_v.cuda(), y_v.cuda()
                    with torch.no_grad():
                        output = self.model(x_v)
                        _, preds = torch.max(output.data, 1)
                        if len(y_v.shape) == 1:
                            val_loss = F.binary_cross_entropy_with_logits(
                                output, y_v.unsqueeze(-1).float(), reduction='mean'
                            ).item()
                        else:
                            val_loss = F.cross_entropy(output,
                                                       y_v.argmax(dim=-1), reduction='mean').item()
                        epoch_val_loss.append(val_loss)
                        _, correct_label = torch.max(y_v.data, 1)
                        iter_corrects = torch.sum(preds == correct_label).to(torch.float32)
                        val_corrects += iter_corrects
                self.val_loss.append(np.mean(epoch_val_loss))

                epoch_acc_val = val_corrects / HAR_VAL_SIZE

                wandb.log({"epoch": epoch + 1, 'loss': round(self.val_loss[-1], 3), 'acc': epoch_acc_val})
                #wandb.watch(self.model)
                #print('Validation accuracy: ', epoch_acc_val)

                print(f'Epoch: {epoch + 1}, '
                     f'Train loss: {round(self.train_loss[-1], 3)}, '
                     f'Val loss: {round(self.val_loss[-1], 3)}, ')

                self.history['train_loss'].append(round(self.train_loss[-1], 3))
                self.history['train_acc'].append(epoch_acc_train)
                self.history['val_loss'].append(round(self.val_loss[-1], 3))
                self.history['val_acc'].append(epoch_acc_val)

                if config.optimizer == "sgd":
                    scheduler.step()

                if self.val_loss[-1] < best_val_loss:
                    best_val_loss = self.val_loss[-1]
                    best_state_dict = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                    if patience_counter == patience:
                        if best_state_dict is not None:
                            self.model.load_state_dict(cast(Dict[str, torch.Tensor], best_state_dict))
                        print('Early stopping!')
                        return None

    def quick_plot_con_matrix(self, y, results, labels):
        # now print confusion metrix
        con = confusion_matrix(y, results)
        conf_matrix = plt.figure(figsize=(25, 15), dpi=50)
        a = sns.heatmap(con, cmap='YlGnBu', annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
        a = plt.rcParams.update({'font.size': 20})
        a = plt.title('Confusion Matrix')
        a = plt.xlabel('Predicted activities')
        a = plt.ylabel('Initial activities')
        plt.show()
        conf_matrix.savefig(('confusion_matrix_' + CURRENT_DATASET + '.png'))

    def quick_plot_con_matrix_normalized(self, y, results, labels):
        # now print confusion metrix
        con = confusion_matrix(y, results, normalize='true')
        conf_matrix = plt.figure(figsize=(25, 15), dpi=50)
        a = sns.heatmap(con, cmap='YlGnBu', annot=True, fmt='.3f', xticklabels=labels, yticklabels=labels)
        a = plt.rcParams.update({'font.size': 20})
        a = plt.title('Confusion Matrix Normalized')
        a = plt.xlabel('Predicted activities')
        a = plt.ylabel('Initial activities')
        plt.show()
        conf_matrix.savefig(('confusion_matrix_normalized_' + CURRENT_DATASET + '.png'))

    def evaluate(self, batch_size: int = 64) -> None:

        test_loader, _ = self.get_loaders(batch_size, mode='test')

        self.model.eval()

        true_list, preds_list = [], []
        for x, y in test_loader:
            #x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                true_list.append(y.detach().numpy())
                preds = self.model(x)
                if len(y.shape) == 1:
                    preds = torch.sigmoid(preds)
                else:
                    preds = torch.softmax(preds, dim=-1)
                preds_list.append(preds.detach().numpy())

        true_np, preds_np = np.concatenate(true_list), np.concatenate(preds_list)
        self.test_results['accuracy_score'] = accuracy_score(
            *self._to_1d_binary(true_np, preds_np)
        )
        print(f'Accuracy score: {round(self.test_results["accuracy_score"], 3)}')

        self.test_results['roc_auc_score'] = roc_auc_score(true_np, preds_np)
        self.test_results['f1_score'] = f1_score(*self._to_1d_binary(true_np, preds_np), average='weighted')
        print(f'ROC AUC score: {round(self.test_results["roc_auc_score"], 3)}')
        print(f'F1 score: {round(self.test_results["f1_score"], 3)}')

        #########################ACC, LOSS GRAPH
        # error_plot = plt.figure(figsize=(12, 8))
        # plt.plot(np.array(self.history['train_loss']), "r--", label="Train loss")
        # plt.plot(np.array(self.history['train_acc']), "r-", label="Train acc")
        # plt.plot(np.array(self.history['val_loss']), "b--", label="Val loss")
        # plt.plot(np.array(self.history['val_acc']), "b-", label="Val acc")
        #
        # plt.title("Training session's progress over iterations")
        # plt.legend(loc='upper right', shadow=True)
        # plt.ylabel('Training Progress (Loss or Accuracy values)')
        # plt.xlabel('Training Epoch')
        # plt.ylim(0)
        # error_plot.savefig('train_val_plot_' + CURRENT_DATASET + '.png')
        # plt.show()

        ########################CONFUSION MATRIX
        # max_test = np.argmax(true_np, axis=1)
        # max_predictions = np.argmax(preds_np, axis=1)
        # activity_map = load_activity_map_wHAR()
        # labels = [activity_map[x] for x in range(len(activity_map))]
        # self.quick_plot_con_matrix_normalized(max_test, max_predictions, labels)
        # self.quick_plot_con_matrix(max_test, max_predictions, labels)

        ########################PLOT ROC CURVE
        # y_true, y_preds = self._to_1d_binary(true_np, preds_np)
        # skplt.metrics.plot_roc(y_true, preds_np)
        # plt.show()

    @staticmethod
    def _to_1d_binary(y_true: np.ndarray, y_preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(y_true.shape) > 1:
            return np.argmax(y_true, axis=-1), np.argmax(y_preds, axis=-1)

        else:
            return y_true, (y_preds > 0.5).astype(int)

    def get_loaders(self, batch_size: int, mode: str,
                    val_size: Optional[float] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Return dataloaders of the training / test data

        Arguments
        ----------
        batch_size:
            The batch size each iteration of the dataloader should return
        mode: {'train', 'test'}
            If 'train', this function should return (train_loader, val_loader)
            If 'test', it should return (test_loader, None)
        val_size:
            If mode == 'train', the fraction of training data to use for validation
            Ignored if mode == 'test'

        Returns
        ----------
        Tuple of (train_loader, val_loader) if mode == 'train'
        Tuple of (test_loader, None) if mode == 'test'
        """
        raise NotImplementedError

    def save_model(self, savepath: Optional[Path] = None) -> Path:
        raise NotImplementedError
