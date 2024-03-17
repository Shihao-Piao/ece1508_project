import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import lr_scheduler
import random
import os
import numpy as np
from collections import OrderedDict
from PIL import Image


def init_model_and_device(
        network: nn.Module,
        use_gpu: bool = True,
        init_scale: float = 0.02
) -> nn.Module:
    """
    Parameters:
        network (nn.Module): The network to initialize and attach to a device.
        use_gpu (bool): Whether to use GPU for training, if available. Defaults to True.
        init_scale (float): The scale factor for initialization, applicable to 'normal' and 'xavier'.
    """

    def init_params_helper(m: nn.Module):
        """Applies the chosen initialization to the model's weights and biases."""
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

            init.normal_(m.weight.data, 0.0, init_scale)

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_scale)
            init.constant_(m.bias.data, 0.0)

    # Determine the device and move the network to it
    device = torch.device("cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu")
    network.to(device)

    # Optionally wrap the network in DataParallel for multi-GPU setups
    if use_gpu and torch.cuda.is_available():
        network = torch.nn.DataParallel(network, device_ids=[0])  # Adjust device_ids as needed

    # Initialize weights and biases
    network.apply(init_params_helper)

    return network



def init_linear_lr(
    optimizer: torch.optim,
    maxsize:int,
    start_epoch: int,
    warmup_epochs: int,
    decay_epochs: int,
    **kwargs
):
    def multiplier(epoch: int):
        # add one to make sure there is never an epoch with 0 lr
        epoch = epoch//maxsize + epoch % maxsize
        lam = 1.0 - max(0.0, start_epoch - 1 + epoch - warmup_epochs) / float(decay_epochs + 1)
        lam = max(lam,0.01)
        return lam
    lr_schedule = lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
    return lr_schedule


class Buffer:
    """
    Implements a buffer for storing tensors, mainly used by the discriminator in CycleGAN architectures.
    This buffer stores a fixed number of tensors to improve training stability.
    """

    def __init__(self, buffer_size: int):
        """
        Initializes the buffer with a specified size.

        Args:
            buffer_size (int): The maximum number of tensors to store in the buffer.
        """
        self.buffer_size = buffer_size
        self.buffer = []
        self.current_count = 0

    def retrieve_tensors(self, input_tensors: "list[torch.Tensor]"):
        """
        Retrieves a batch of tensors from the buffer, replacing them with new input tensors at a chance.

        Args:
            input_tensors (list[torch.Tensor]): A list of new fake tensors to potentially add to the buffer.

        Returns:
            torch.Tensor: A batch of tensors for the discriminator to use, composed of tensors from the buffer
                          and/or the input tensors, depending on the buffer's current state and policies.
        """
        if self.buffer_size <= 0:
            return torch.cat(input_tensors, dim=0)

        returned_tensors = []
        for tensor in input_tensors:
            tensor = tensor.detach()
            if self.current_count < self.buffer_size:
                self.buffer.append(tensor)
                self.current_count += 1
                returned_tensors.append(tensor)
            else:
                # Decide whether to use a tensor from the buffer or the new tensor
                if random.random() < 0.5:
                    # Randomly replace and return a tensor from the buffer
                    index = random.randint(0, self.buffer_size - 1)
                    returned_tensors.append(self.buffer[index].clone())
                    self.buffer[index] = tensor
                else:
                    # Use the new tensor
                    returned_tensors.append(tensor)

        return torch.cat(returned_tensors, dim=0)


def print_losses(losses, epoch, total_epochs, iters, total_iters, subset=None):
    """
    Prints the specified losses at the end of each dataloader iteration.

    Parameters:
        losses (OrderedDict): An ordered dictionary containing loss names and values.
        epoch (int): The current training epoch.
        total_epochs (int): The total number of epochs for training.
        iters (int): The current iteration within the current epoch.
        total_iters (int): The total number of iterations within the current epoch.
        subset (list, optional): A subset of keys from the losses to print. If None, all losses are printed.
    """
    # just print the total loss, can adjust
    key = 'total_loss'
    loss_text = " ".join([f"{key}: {losses[key]:.7f}"])
    print(f"[{epoch}/{total_epochs}] [{iters}/{total_iters}] Losses: {loss_text}")

    '''
    subset = subset if subset is not None else losses.keys()
    loss_text = " ".join([f"{key}: {losses[key]:.7f}" for key in subset])
    print(f"[{epoch}/{total_epochs}] [{iters}/{total_iters}] Losses: {loss_text}")
    '''

def get_latest_num(checkpoints_dir):
    """
    Retrieves the epoch number of the latest checkpoint saved.

    Parameters:
        checkpoints_dir (str): The directory containing the checkpoints.

    Returns:
        int: The epoch number of the latest checkpoint. Returns -1 if no checkpoint is found.
    """
    epoch_numbers = [
        int(filename.split('_')[0]) for filename in os.listdir(checkpoints_dir)
        if filename.endswith('.pth') and filename.split('_')[0].isdigit()
    ]
    return max(epoch_numbers, default=-1)

def save_outs(outs, out_dir, save_separate=False, extension='jpg', save=True):
    """
    Saves images contained within `outs` either as a combined image or separately.

    Parameters:
        outs (OrderedDict): An ordered dictionary containing image tensors.
        out_dir (str): The directory to save the images.
        save_separate (bool, optional): Whether to save each image separately.
        extension (str, optional): The file extension for the saved images.
        save (bool, optional): If False, skips the saving process.

    Returns:
        np.array: The combined image array, mainly for testing purposes.
    """
    if not save:
        return None

    os.makedirs(out_dir, exist_ok=True)
    images = []

    for img_tensor in outs.values():
        img = img_tensor.cpu().float().numpy()
        img = np.transpose(img, (0, 2, 3, 1))  # Convert from NCHW to NHWC
        img = np.tile(img, (1, 1, 1, 3)) if img.shape[-1] == 1 else img  # Grayscale to RGB
        img = ((img + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
        images.append(img)

    combined_image = np.concatenate(images, axis=2)[0]

    if save:
        Image.fromarray(combined_image).save(os.path.join(out_dir, f"combined.{extension}"))
        if save_separate:
            for name, img in zip(outs.keys(), images):
                Image.fromarray(img[0]).save(os.path.join(out_dir, f"{name}.{extension}"))

    return combined_image


