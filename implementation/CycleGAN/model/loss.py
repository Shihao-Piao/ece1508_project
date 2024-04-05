import torch.nn as nn
import torch


class Loss(nn.Module):
    """
    Initializes a customizable Loss class for training models.
    """

    def __init__(
            self,
            loss_type: str = 'mse',
            target_real_label: float = 1.0,
            target_fake_label: float = 0.0
    ):
        """
        Constructs the CustomLoss class with specified parameters.

        Args:
            loss_type (str): The type of loss function to use, defaults to 'mse'.
            target_real_label (float): The label value for real data, defaults to 1.0.
            target_fake_label (float): The label value for fake data, defaults to 0.0.
        """
        super(Loss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss_type = loss_type
        self.loss = nn.MSELoss()

    def prepare_labels(self, predictions: torch.Tensor, is_real: bool):
        """
        Generates the appropriate labels for the loss calculation.

        Args:
            predictions (torch.Tensor): The predictions made by the model.
            is_real (bool): Flag to indicate whether the labels should be real or fake.

        Returns:
            torch.Tensor: A tensor of labels matching the predictions' shape.
        """
        if is_real:
            return self.real_label.expand_as(predictions)
        return self.fake_label.expand_as(predictions)

    def __call__(self, predictions: torch.Tensor, is_real: bool):
        """
        Calculates and returns the loss when the instance is called.

        Args:
            predictions (torch.Tensor): The predictions made by the model.
            is_real (bool): Specifies whether to use real or fake labels for loss calculation.

        Returns:
            torch.Tensor: The calculated loss value.
        """
        labels = self.prepare_labels(predictions, is_real)
        return self.loss(predictions, labels)
