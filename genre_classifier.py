import pickle
import os
import torch
import typing as tp
from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt
import librosa.feature
import librosa
import librosa.display
import torchaudio

from torch.utils.data import DataLoader
from evaluate import evaluate_model, get_model_accuracy, plot_loss
from data_loader import DataSet
from feature_cache import FeatureCache
from utils import Genre

plt.style.use('ggplot')


@dataclass
class TrainingParameters:
    """
    This dataclass defines a training configuration.
    feel free to add/change it as you see fit, do NOT remove the following fields as we will use
    them in test time.
    If you add additional values to your training configuration please add them in here with
    default values (so run won't break when we test this).
    """

    batch_size: int = 256
    num_epochs: int = 150
    train_json_path: str = "jsons/train.json"  # you should use this file path to load your train data
    test_json_path: str = "jsons/test.json"  # you should use this file path to load your test data

    save_dir: str = 'model_files'


@dataclass
class OptimizationParameters:
    """
    This dataclass defines optimization related hyper-parameters to be passed to the model.
    feel free to add/change it as you see fit.
    """

    learning_rate: float = 0.001

    num_of_features: int = 45771
    num_of_genre: int = 3
    eval_every: int = 10
    sample_rate: int = 22050

    dropout_rate = 0.85
    regularization_factor = 0.001


class MusicClassifier:
    """
    You should Implement your classifier object here
    """

    def __init__(self, opt_params: OptimizationParameters, **kwargs):
        """
        This defines the classifier object.
        - You should defiend your weights and biases as class components here.
        - You could use kwargs (dictionary) for any other variables you wish to pass in here.
        - You should use `opt_params` for your optimization and you are welcome to experiment
        """
        self.opt_params = opt_params
        self.W = torch.rand((opt_params.num_of_features, opt_params.num_of_genre))
        self.b = torch.rand(opt_params.num_of_genre)

    @staticmethod
    def extract_amplitude_envelope(one_wav):
        hop_length = 256
        amplitude_envelope = []
        for i in range(0, len(one_wav), hop_length):
            amplitude_envelope.append(np.abs(one_wav[i: i + hop_length]).max())
        return torch.tensor(amplitude_envelope)

    def extract_feats(self, wavs: torch.Tensor):
        """
        this function extract features from a given audio.
        we will not be observing this method.
        """
        return torch.stack([self.extract_feats_one_wav(one_wav) for one_wav in wavs])

    def extract_feats_one_wav(self, one_wav: torch.Tensor):

        mfcc = torchaudio.transforms.MFCC(sample_rate=self.opt_params.sample_rate, n_mfcc=40, melkwargs={"n_fft": 512})(one_wav).numpy().squeeze()
        one_wav = one_wav.numpy().squeeze()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=one_wav).squeeze()
        tempo = librosa.beat.tempo(y=one_wav, sr=self.opt_params.sample_rate).flatten()
        stats_feats = [one_wav.std(), one_wav.mean()] + [np.percentile(one_wav, i) for i in range(0, 101, 10)]
        mfcc_stats = [mfcc.std(axis=1), mfcc.mean(axis=1)] + [np.percentile(mfcc, i, axis=1) for i in range(0, 101, 10)]
        amplitude_envelope = self.extract_amplitude_envelope(one_wav)
        amplitude_envelope_first_der = np.diff(amplitude_envelope)
        amplitude_envelope_sec_der = np.diff(amplitude_envelope_first_der)

        features = torch.Tensor(
            np.concatenate(
                [
                    np.array(mfcc_stats).flatten(),
                    np.array(stats_feats),
                    mfcc.flatten(),
                    zero_crossing_rate,
                    tempo,
                    amplitude_envelope,
                    amplitude_envelope_first_der,
                    amplitude_envelope_sec_der,
                ]
            )
        )

        assert len(features) == self.opt_params.num_of_features, f'new num_of_features {len(features)}'
        return features

    def forward(self, feats: torch.Tensor, in_train=True) -> tp.Any:
        """
        this function performs a forward pass throuh the model, outputting scores for every class.
        feats: batch of extracted faetures
        """
        if in_train:
            dropout_mask = torch.rand(feats.shape) > self.opt_params.dropout_rate
            feats = feats * dropout_mask
        else:
            feats = feats * (1 - self.opt_params.dropout_rate)
        model_output = torch.matmul(feats, self.W) + self.b

        def softmax(x):
            e_x = torch.exp(x - x.max(dim=-1).values.unsqueeze(-1))
            return e_x / e_x.sum(dim=-1).unsqueeze(-1)

        return softmax(model_output)

    def backward(
            self,
            feats: torch.Tensor,
            y_pred: torch.Tensor,
            labels: torch.Tensor,
            train=True,
    ):
        """
        this function should perform a backward pass through the model.
        - calculate loss
        - calculate gradients
        - update gradients using SGD

        Note: in practice - the optimization process is usually external to the model.
        We thought it may result in less coding needed if you are to apply it here, hence
        OptimizationParameters are passed to the initialization function
        """
        labels_one_hot = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=self.opt_params.num_of_genre)
        y_pred_argmax = torch.argmax(y_pred, dim=-1)

        # L2 loss
        loss = ((labels_one_hot - y_pred) ** 2).mean()
        acc = get_model_accuracy(labels, y_pred_argmax)

        if not train:
            return loss, acc

        # calculate gradients
        batch_size = feats.shape[0]

        diff = y_pred - labels_one_hot

        dW = feats.T.matmul(diff) / batch_size
        db = torch.sum(diff, dim=0) / batch_size

        dW += 2 * self.opt_params.regularization_factor * self.W

        # update weights
        self.W -= self.opt_params.learning_rate * dW
        self.b -= self.opt_params.learning_rate * db

        return loss, acc

    def get_weights_and_biases(self) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        This function returns the weights and biases associated with this model object,
        should return a tuple: (weights, biases)
        """
        return self.W, self.b

    def classify(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        this method should recieve a torch.Tensor of shape [batch, channels, time] (float tensor)
        and a output batch of corresponding labels [B, 1] (integer tensor)
        """
        features = self.extract_feats(wavs)
        genre_score = self.forward(features, in_train=False)
        predicted_labels = torch.argmax(genre_score, dim=-1, keepdim=True)
        return predicted_labels

    def save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.W, os.path.join(save_dir, 'W.pt'))
        torch.save(self.b, os.path.join(save_dir, 'b.pt'))

    def load_model(self, load_dir):
        self.W = torch.load(os.path.join(load_dir, 'W.pt'))
        self.b = torch.load(os.path.join(load_dir, 'b.pt'))


class ClassifierHandler:
    @staticmethod
    def train_new_model(training_parameters: TrainingParameters, use_pickle: bool = False) -> MusicClassifier:
        """
        This function should create a new 'MusicClassifier' object and train it from scratch.
        You could program your training loop / training manager as you see fit.
        """
        opt_params = OptimizationParameters()

        model = MusicClassifier(opt_params)

        if use_pickle and os.path.exists('feature_cache.pkl'):
            feature_cache = pickle.load(open('feature_cache.pkl', 'rb'))
            print(f'load feature cache from pickle: {len(feature_cache.cache)}!!!')
            print('Warning: feature cache is not updated!!!')
        else:
            feature_cache = FeatureCache(model)

        train_dataset = DataSet(json_dir=training_parameters.train_json_path, feature_cache=feature_cache)
        train_loader = DataLoader(train_dataset, batch_size=training_parameters.batch_size, shuffle=True)

        test_dataset = DataSet(json_dir=training_parameters.test_json_path, feature_cache=feature_cache)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        test_features, test_labels = next(test_loader.__iter__())

        test_losses = []
        test_epochs = []
        train_losses = []
        train_epochs = []
        best_score = -float('inf')
        for epoch_num in range(training_parameters.num_epochs):
            loss_mean = 0
            acc_mean = 0
            for features, labels in train_loader:
                y_pred = model.forward(features, in_train=True)
                loss, acc = model.backward(features, y_pred, labels)
                loss_mean += loss / len(train_loader)
                acc_mean += acc / len(train_loader)

            if epoch_num == 0:
                with open('feature_cache.pkl', 'wb') as f:
                    pickle.dump(feature_cache, f)
                print(f'save feature cache to pickle: {len(feature_cache.cache)}!!!')
            print(f'Train - epoch_num: {epoch_num}, loss: {loss_mean:.3f}, acc: {acc_mean:.3f}')
            train_losses.append((loss_mean, acc_mean))
            train_epochs.append(epoch_num)

            # test
            if (epoch_num + 1) % opt_params.eval_every == 0:
                test_pred = model.forward(test_features, in_train=False)
                loss_mean, acc = model.backward(test_features, test_pred, test_labels, train=False)
                print(f'Test - epoch_num: {epoch_num}, loss: {loss_mean:.3f}, acc: {acc:.3f}')
                test_losses.append((loss_mean, acc))
                test_epochs.append(epoch_num)
                if acc > best_score:
                    best_score = acc
                    model.save_model(training_parameters.save_dir)
                    print(f'Model saved to: {training_parameters.save_dir}')

        model.load_model(training_parameters.save_dir)
        plot_loss(test_epochs, test_losses, train_epochs, train_losses, training_parameters)
        genre_score = model.forward(test_features, in_train=False)
        test_pred = torch.argmax(genre_score, dim=-1)
        evaluate_model(test_labels, test_pred, training_parameters.save_dir)
        return model

    @staticmethod
    def get_pretrained_model(dir_path='model_files') -> MusicClassifier:
        """
        This function should construct a 'MusicClassifier' object, load it's trained weights /
        hyperparameters and return the loaded model
        """
        opt_params = OptimizationParameters()
        model = MusicClassifier(opt_params)
        model.load_model(dir_path)
        return model


if __name__ == '__main__':
    trains_params = TrainingParameters()
    model = ClassifierHandler.train_new_model(trains_params, use_pickle=True)
