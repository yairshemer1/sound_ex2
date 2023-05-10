from abc import abstractmethod
import torch
from enum import Enum
import typing as tp
from dataclasses import dataclass
import numpy as np


class Genre(Enum):
    """
    This enum class is optional and defined for your convinience, you are not required to use it.
    Please use the int labels this enum defines for the corresponding genras in your predictions.
    """
    CLASSICAL: int = 0
    HEAVY_ROCK: int = 1
    REGGAE: int = 2


@dataclass
class TrainingParameters:
    """
    This dataclass defines a training configuration.
    feel free to add/change it as you see fit, do NOT remove the following fields as we will use
    them in test time.
    If you add additional values to your training configuration please add them in here with 
    default values (so run won't break when we test this).
    """
    batch_size: int = 32
    num_epochs: int = 100
    train_json_path: str = "jsons/train.json"  # you should use this file path to load your train data
    test_json_path: str = "jsons/test.json"  # you should use this file path to load your test data


@dataclass
class OptimizationParameters:
    """
    This dataclass defines optimization related hyper-parameters to be passed to the model.
    feel free to add/change it as you see fit.
    """
    learning_rate: float = 0.001

    # num_of_features: int = 1024
    num_of_features: int = 2
    num_of_genre: int = 3


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

    def exctract_feats(self, wavs: torch.Tensor):
        """
        this function extract features from a given audio.
        we will not be observing this method.
        """
        # Suffel
        # Ufmentaiuon
        # norm
        raise NotImplementedError("optional, function is not implemented")

    def forward(self, feats: torch.Tensor) -> tp.Any:
        """
        this function performs a forward pass throuh the model, outputting scores for every class.
        feats: batch of extracted faetures
        """
        model_output = torch.matmul(feats, self.W) + self.b

        def softmax(x):
            e_x = torch.exp(x - x.max(dim=-1).values.unsqueeze(-1))
            return e_x / e_x.sum(dim=-1).unsqueeze(-1)

        return softmax(model_output)

    def backward(self, feats: torch.Tensor, output_scores: torch.Tensor, labels: torch.Tensor):
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
        y_pred = self.forward(feats)

        # L2 loss
        loss = ((labels_one_hot - y_pred) ** 2).mean()

        # calculate gradients
        batch_size = feats.shape[0]
        dW = - 2 * feats.T.matmul(labels_one_hot - y_pred) / batch_size
        db = - 2 * torch.sum(labels_one_hot - y_pred, dim=0) / batch_size

        # update weights
        self.W = self.W - self.opt_params.learning_rate * dW
        self.b = self.b - self.opt_params.learning_rate * db

        return loss

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
        # features = self.exctract_feats(wavs)
        genre_score = self.forward(wavs)
        predicted_labels = torch.argmax(genre_score, dim=-1)
        return predicted_labels


class ClassifierHandler:

    @staticmethod
    def train_new_model(training_parameters: TrainingParameters) -> MusicClassifier:
        """
        This function should create a new 'MusicClassifier' object and train it from scratch.
        You could program your training loop / training manager as you see fit.
        """
        raise NotImplementedError("function is not implemented")

    @staticmethod
    def get_pretrained_model() -> MusicClassifier:
        """
        This function should construct a 'MusicClassifier' object, load it's trained weights / 
        hyperparameters and return the loaded model
        """
        raise NotImplementedError("function is not implemented")



def creat_dummy_data():
    # Define the means and covariance matrices for each of the three Gaussians
    mean1 = torch.Tensor([1, 1])
    cov1 = torch.Tensor([[0.1, 0], [0, 0.1]])

    mean2 = torch.Tensor([-1, -1])
    cov2 = torch.Tensor([[0.1, 0], [0, 0.1]])

    mean3 = torch.Tensor([1, -1])
    cov3 = torch.Tensor([[0.1, 0], [0, 0.1]])

    # Generate 100 samples from each of the Gaussians
    num_samples = 100
    gaussian1_samples = torch.randn(num_samples, 2) @ torch.diag(torch.sqrt(cov1)) + mean1.unsqueeze(-1)
    gaussian2_samples = torch.randn(num_samples, 2) @ torch.diag(torch.sqrt(cov2)) + mean2.unsqueeze(-1)
    gaussian3_samples = torch.randn(num_samples, 2) @ torch.diag(torch.sqrt(cov3)) + mean3.unsqueeze(-1)

    # Concatenate the samples and labels
    X = torch.cat((gaussian1_samples, gaussian2_samples, gaussian3_samples), dim=-1)
    y = torch.cat((torch.zeros(num_samples), torch.ones(num_samples), torch.ones(num_samples) * 2))

    # Shuffle the data
    perm = torch.randperm(num_samples * 3)
    X = X[:, perm]
    y = y[perm]

    return X, y

if __name__ == '__main__':
    params = OptimizationParameters()
    trains_params = TrainingParameters()
    model = MusicClassifier(params)

    X, y = creat_dummy_data()
    X = X.T
    # f = torch.rand((trains_params.batch_size, params.num_of_features))
    # l = torch.randint(0, 3, size=(trains_params.batch_size, ))
    losses = [model.backward(X, y, y) for _ in range(10000)]
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(losses)), losses)
    plt.show()

    X, y = creat_dummy_data()
    X = X.T
    print((model.classify(X) == y).type(torch.float).mean())