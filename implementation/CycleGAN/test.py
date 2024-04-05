import torch
import os
from model.CycleGAN import CycleGAN
from data.dataset import CycleDataset
from utils import *

import warnings
warnings.filterwarnings("ignore")

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
        "checkpoints_dir": "checkpoints",
        "model_name": "my_CycleGAN",
        "use_gpu": False,  # change to True
        "to_train": False,
        "load_epoch": 'latest',
        "continue_train": False,
        "result_dir" : 'results'
    }

    # Override some config settings for testing phase
    config['phase'] = 'test'
    result_dir = os.path.join(config['result_dir'], f"{config['model_name']}_{config['phase']}")
    os.makedirs(result_dir, exist_ok=True)
    config['dataset']['scale_size'] = config['dataset']['crop_size']  # Adjusting scale size to match crop size for testing

    dataset = CycleDataset(False, 'data', **config['dataset'])  # False indicates not training
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config['dataset']['num_workers'])

    max_size = len(dataloader)
    # Initialize model and dataset
    model = CycleGAN(config)
    model.general_setup(max_size)

    
    print(f"Saving images in {result_dir}")
    num_tests = config.get('num_tests', len(dataloader))  # Use config or default to the size of the dataloader

    for i, data in enumerate(dataloader):
        if i >= num_tests:
            break
        model.setup_input(data)
        out = model.test()  # Acquiring the test output

        # Construct the output file name based on the input image path
        file_name = os.path.splitext(os.path.basename(data['X_paths'][0]))[0]
        save_outs(out, os.path.join(result_dir, file_name), config.get('save_separate', False))