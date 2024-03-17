import torch
import os
from model.CycleGAN import CycleGAN
from data.dataset import CycleDataset
from utils import *
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings("ignore")

def save_losses_as_pkl(losses, filename):
    """
    Saves the losses dictionary as a pickle file.

    Parameters:
        losses (dict): A dictionary containing the loss values to save.
        filename (str): The filename for the saved pickle file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(losses, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    config = {
        "dataset": {
            "batch_size": 1,
            "in_order": False,
            "scale_size": 286,
            "crop_size": 256,
            "in_channels": 3,
            "out_channels": 3,
            "num_workers": 4
        },
        "model": {
            "generator": {
                "num_filters": 64,
                "num_blocks": 9,
                "num_sampling": 2,
                "use_dropout": False,
                "init_scale": 0.02
            },
            "discriminator": {
                "num_filters": 64,
                "num_conv_layers": 3,
                "ker_size": 4,
                "padding": 1,
                "init_scale": 0.02
            }
        },
        "train": {
            "save_epoch_freq": 10,
            "warmup_epochs": 70, #70
            "decay_epochs": 30, #30
            "beta1": 0.5,
            "lr": 0.0002,
            "buffer_size": 50,
            "loss": {
                "loss_type": 'mse',
                "lambda_scaling": 0.5,
                "lambda_X": 10.0,
                "lambda_Y": 10.0
            }
        },
        "checkpoints_dir": "checkpoints",
        "model_name": "my_CycleGAN",
        "use_gpu": True,  # change to True
        "to_train": True,
        "load_epoch": 'latest',
        'continue_train': True
    }

    # Extract training parameters from config
    warmup_epochs = config['train']['warmup_epochs']
    decay_epochs = config['train']['decay_epochs']
    save_epoch_freq = config['train']['save_epoch_freq']
    total_epochs = warmup_epochs + decay_epochs
    checkpoints_dir = config['checkpoints_dir']
    model_name = config['model_name']
    continue_train = config.get('continue_train', False)
    load_epoch = config.get('load_epoch', 'latest')

    # Determine the starting epoch
    if not continue_train:
        start_epoch = 1
    else:
        if load_epoch == 'latest':
            start_epoch = get_latest_num(os.path.join(checkpoints_dir, model_name)) + 1
        else:
            start_epoch = int(load_epoch) + 1
    config['train']['start_epoch'] = start_epoch

    # Create model and dataset instances
    dataset_config = config['dataset']
    dataset = CycleDataset(True, 'data/',**dataset_config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset_config['batch_size'],
                                             shuffle=not dataset_config['in_order'],
                                             num_workers=dataset_config['num_workers'])

    # Training preparation
    max_size = len(dataloader)
    model = CycleGAN(config)
    model.general_setup(max_size)
    X_size, Y_size = dataset.both_len()
    print(f'Number of X images: {X_size}, Number of Y images: {Y_size}')
    print(f"Starting training loop from epoch {start_epoch}...")
    print("Losses printed as [epoch / total epochs] [batch / total batches]")

    # Training loop
    print("Total epochs: ", total_epochs)
    epoch_losses = []
    for epoch in range(start_epoch, total_epochs + 1):
        print(f"LR: {next(iter(model.schedulers)).get_last_lr()}")  # Assumes model.schedulers is a list of scheduler
        epoch_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
        for i, data in enumerate(epoch_bar):
            model.setup_input(data)
            model.optimize()
            model.update_schedulers()
            losses = model.get_losses()  # Ordered dict
            #tqdm.write(f"Epoch [{epoch}/{total_epochs}] - Batch [{i+1}/{len(dataloader)}]: {losses}")
            #print_losses(losses, epoch, total_epochs, i + 1, max_size)
            
        epoch_losses.append(losses)
        print_losses(losses, epoch, total_epochs, max_size, max_size)
        if (epoch % save_epoch_freq == 0) or (epoch == total_epochs):
            print(f"Saving models at end of epoch {epoch}")
            model.save_networks()
            model.save_networks(str(epoch))
            save_losses_as_pkl(epoch_losses,'checkpoints/loss.pkl')