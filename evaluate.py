import numpy as np
import typing as tp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import torch
import os
from utils import Genre


def evaluate_model(y_true: torch.Tensor, y_pred: torch.Tensor, save_dir):
    # calculate accuracy
    y_true, y_pred = y_true.numpy(), y_pred.numpy()
    num_correct = np.sum(y_true == y_pred)
    accuracy = num_correct / len(y_true)
    # calculate recall
    num_classes = len(Genre)
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        confusion_matrix[y_true[i], y_pred[i]] += 1
    for i in range(num_classes):
        confusion_matrix[i] /= np.sum(confusion_matrix[i])
    recall = np.mean([confusion_matrix[i, i] / np.sum(confusion_matrix[i, :]) for i in range(num_classes)])
    # plot confusion matrix
    df_cm = pd.DataFrame(confusion_matrix, index=[i for i in Genre], columns=[i for i in Genre])
    plt.figure(figsize=(10, 7))
    ax = sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='.2f')
    ax.xaxis.tick_top()
    plt.title(f"Accuracy: {accuracy:.2f}, Recall: {recall:.2f}")
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.show()


def get_model_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    num_correct = torch.sum(y_true == y_pred)
    accuracy = num_correct / len(y_true)
    return accuracy

def plot_loss(test_epochs, test_losses, train_epochs, train_losses, training_parameters):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(train_epochs, [loss for loss, _ in train_losses], label='train')
    axs[0].plot(test_epochs, [loss for loss, _ in test_losses], label='test')
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('loss')
    axs[0].legend()
    axs[1].plot(train_epochs, [acc for _, acc in train_losses], label='train')
    axs[1].plot(test_epochs, [acc for _, acc in test_losses], label='test')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('acc')
    axs[1].legend()
    plt.savefig(os.path.join(training_parameters.save_dir, 'loss_graph.png'))
    plt.show()


