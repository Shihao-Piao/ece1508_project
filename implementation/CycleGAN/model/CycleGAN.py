from argparse import Namespace
import sys
sys.path.append('..')
import itertools
import os
import torch
import torch.nn as nn
#from model.buffer import Buffer
#from utils.model_utils import print_network
from collections import OrderedDict
from utils import *
from model.block import *

class Loss(nn.Module):
    """
    Initializes a customizable Loss class for training models.
    """

    def __init__(self,loss_type = 'mse',target_real_label = 1.0,target_fake_label = 0.0):
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

    def prepare_labels(self, predictions, is_real):
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

    def __call__(self, predictions, is_real):
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

class CycleGAN(nn.Module):
    def __init__(self, config: dict):
        """
        Parameters:
            config (dict) - A configuration dictionary specifying the
                            network architectures, directories, device preferences,
                            and any other options required for instantiation.
        """
        super(CycleGAN, self).__init__()
        self.config = config
        self.to_train = config.get("to_train", True)

        # Create log directory
        self.save_dir = os.path.join(config['checkpoints_dir'], config['model_name'])
        print(f"Saving logs in {self.save_dir}")
        os.makedirs(self.save_dir, exist_ok=True)

        # Device configuration
        use_gpu = config.get('use_gpu', False)
        self.device = torch.device('cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu')
        if use_gpu:
            torch.cuda.set_device(0)

        # Define nets
        self.in_channels = config['dataset']['in_channels']
        self.out_channels = config['dataset']['out_channels']

        self.genG = Generator(self.in_channels,self.out_channels)
        self.genG = init_model_and_device(self.genG)

        self.genF = Generator(self.in_channels,self.out_channels)
        self.genF = init_model_and_device(self.genF)

        self.model_names = ['genG', 'genF']

        if self.to_train:
            # Discriminator for domain X (D_X)
            self.netD_X = Discriminator(self.in_channels)
            self.netD_X = init_model_and_device(self.netD_X)

            # Discriminator for domain Y (D_Y)
            self.netD_Y = Discriminator(self.out_channels)
            self.netD_Y = init_model_and_device(self.netD_Y)

            self.model_names.extend(['netD_X', 'netD_Y'])

            # Define loss functions/objective functions
            self.loss_func = Loss().to(self.device)
            self.loss_cycle = torch.nn.L1Loss().to(self.device)
            self.loss_identity = torch.nn.L1Loss().to(self.device)

            # Define image pools
            self.buffer_size = config['train']['buffer_size']
            self.fake_X_buffer = Buffer(self.buffer_size)
            self.fake_Y_buffer = Buffer(self.buffer_size)

            # Define optimizers and learning rate schedulers
            lr = config['train']['lr']
            beta1 = config['train']['beta1']
            self.optim_G = torch.optim.Adam(
                itertools.chain(self.genG.parameters(), self.genF.parameters()),
                lr=lr, betas=(beta1, 0.999)
            )
            self.optim_D = torch.optim.Adam(
                itertools.chain(self.netD_X.parameters(), self.netD_Y.parameters()),
                lr=lr, betas=(beta1, 0.999)
            )
            self.optimizers = [self.optim_G, self.optim_D]

    def setup_input(self, input: dict):
        """
        Prepares the input data for processing by the networks.
        """
        self.real_X = input['X'].to(self.device)  # Input images from domain X
        self.real_Y = input['Y'].to(self.device)  # Input images from domain Y
        self.image_paths = input.get('X_paths')  # Paths for images from domain X
        self.y_paths = input.get('Y_paths')  # Paths for images from domain Y

    def forward(self):
        # Generate fake Y from real X and vice versa
        self.fake_Y = self.genG(self.real_X)
        self.fake_X = self.genF(self.real_Y)

        # Generate reconstructions by transforming the fake images back to the original domain
        self.recon_X = self.genF(self.fake_Y)  # Attempt to reconstruct X -> G(X) -> F(G(X))
        self.recon_Y = self.genG(self.fake_X)  # Attempt to reconstruct Y -> F(Y) -> G(F(Y))

    def backward_D(self, discriminator, real, fake, factor = 0.5):
        """
        Computes the loss for a given discriminator, combining the losses
        from real and fake images and performing a backward pass.

        Parameters:
            discriminator (nn.Module): The discriminator network for which to compute the loss.
            real (torch.Tensor): A batch of real images.
            fake (torch.Tensor): A batch of fake images generated by the generators.
            factor (float): A scaling factor for the loss, allowing for adjustment of the discriminator's influence.

        Returns:
            torch.Tensor: The computed loss value for the discriminator.
        """
        # Discriminator predictions for real images
        pred_real = discriminator(real)
        # Discriminator predictions for fake images, detached to prevent gradients from flowing into the generator
        pred_fake = discriminator(fake.detach())

        # Calculate losses for real and fake predictions
        loss_real = self.loss_func(pred_real, True)
        loss_fake = self.loss_func(pred_fake, False)

        # Combine losses and apply scaling factor
        total_loss = (loss_real + loss_fake) * factor
        total_loss.backward()

        return total_loss

    def backward_D_X(self):
        """
        Computes the loss and updates gradients for the discriminator D_X. This discriminator
        differentiates between real images from domain X and fake images generated from domain Y.
        Fake images are retrieved from a buffer that mixes recent and older generated images to
        improve training stability.
        """
        # Retrieve fake images from the buffer to introduce historical generated images
        buffered_fake_X = self.fake_X_buffer.retrieve_tensors(self.fake_X)
        # Calculate the loss for discriminator D_X with both real and buffered fake images
        # and backpropagate the gradients
        self.loss_D_X = self.backward_D(discriminator=self.netD_X, real=self.real_X, fake=buffered_fake_X)

    def backward_D_Y(self):
        """
        Computes the loss and updates gradients for the discriminator D_Y. This discriminator
        differentiates between real images from domain Y and fake images generated from domain X.
        Utilizes a buffer for fake images to incorporate a mix of recent and older images,
        aiding in preventing discriminator overfitting.
        """
        # Retrieve fake images from the buffer for a varied training sample
        buffered_fake_Y = self.fake_Y_buffer.retrieve_tensors(self.fake_Y)
        # Calculate the loss for discriminator D_Y using real images and buffered fake images,
        # then backpropagate the gradients
        self.loss_D_Y = self.backward_D(discriminator=self.netD_Y, real=self.real_Y, fake=buffered_fake_Y)

    def backward_G(self):
        """
        Computes and backpropagates the total loss for the generators, including adversarial,
        cycle consistency, and optional identity loss components.
        """
        # Configurable weighting factors for different loss components
        lambda_X = self.config['train']['loss']['lambda_X']
        lambda_Y = self.config['train']['loss']['lambda_Y']
        lambda_identity = self.config['train']['loss'].get('lambda_scaling', 0)

        # Identity Loss
        self.identity_loss_G = 0
        if lambda_identity > 0:
            self.identity_loss_G = (
                                           self.loss_identity(self.genG(self.real_Y), self.real_Y) * lambda_Y +
                                           self.loss_identity(self.genF(self.real_X), self.real_X) * lambda_X
                                   ) * lambda_identity

        # Adversarial Loss (G tries to fool D into thinking the generated images are real)
        self.loss_G_adv_X = self.loss_func(self.netD_X(self.fake_X), True)
        self.loss_G_adv_Y = self.loss_func(self.netD_Y(self.fake_Y), True)

        # Cycle Consistency Loss
        self.loss_cycle_X = self.loss_cycle(self.recon_X, self.real_X) * lambda_X
        self.loss_cycle_Y = self.loss_cycle(self.recon_Y, self.real_Y) * lambda_Y

        # Total Generator Loss
        self.loss_G_total = self.loss_G_adv_X + self.loss_G_adv_Y + self.loss_cycle_X + self.loss_cycle_Y + self.identity_loss_G
        self.loss_G_total.backward()

    def optimize(self):
        """
        Executes one training step: performs a forward pass, then updates both the
        generators and discriminators based on their respective losses.
        """
        # Forward pass through the network
        self.forward()

        # Update Generators
        self.set_requires_grad([self.netD_X, self.netD_Y], False)  # Freeze discriminators
        self.optim_G.zero_grad()  # Reset gradients for generator optimizers
        self.backward_G()  # Compute gradients for generators
        self.optim_G.step()  # Apply gradients and update generator weights

        # Update Discriminators
        self.set_requires_grad([self.netD_X, self.netD_Y], True)  # Unfreeze discriminators
        self.optim_D.zero_grad()  # Reset gradients for discriminator optimizers
        self.backward_D_X()  # Compute gradients for D_X
        self.backward_D_Y()  # Compute gradients for D_Y
        self.optim_D.step()  # Apply gradients and update discriminator weights

    def set_requires_grad(self, nets, requires_grad):
        """
        Sets the .requires_grad attribute of the parameters in the given networks.
        Parameters:
            nets (list): A list of networks whose parameters' .requires_grad flags will be set.
            requires_grad (bool): The desired state of the .requires_grad flag.
        """
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def general_setup(self,maxsize):
        """
        Prepares the model for training or evaluation based on the configuration.
        """
        if self.to_train:
            self.setup_schedulers(maxsize)
        if not self.to_train or self.config['continue_train']:
            self.load_networks(self.config['load_epoch'])
        #print_network(self, verbose=self.opt.verbose)

    def setup_schedulers(self,maxsize):
        """
        Initializes learning rate schedulers for each optimizer.
        """
        self.schedulers = [init_linear_lr(optimizer,maxsize = maxsize, **self.config['train']) for optimizer in self.optimizers]


    def update_schedulers(self):
        """
        Advances the learning rate schedulers to the next step.
        """
        for scheduler in self.schedulers:
            scheduler.step()

    def save_networks(self, epoch = 'latest'):
        """
        Saves each network's state dictionary for the specified epoch.

        Parameters:
            epoch (str): Identifier for the save epoch, defaults to 'latest'.
        """
        for name in self.model_names:
            file_name = f'{epoch}_{name}.pth'
            destination = os.path.join(self.save_dir, file_name)
            model = getattr(self, name, None)
            if model:
                torch.save(model.state_dict(), destination)

    def load_networks(self, epoch = 'latest'):
        """
        Loads network weights from the specified epoch.

        Parameters:
            epoch (str): The epoch identifier for loading the models, defaults to 'latest'.
        """
        for model_name in self.model_names:
            load_path = os.path.join(self.save_dir, f'{epoch}_{model_name}.pth')
            net = getattr(self, model_name)
            state_dict = torch.load(load_path, map_location=self.device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict)

    def eval(self):
        """Sets all networks to evaluation mode."""
        for model_name in self.model_names:
            net = getattr(self, model_name, None)
            if net:
                net.eval()

    def test(self, to_return=['real_X', 'fake_Y']):
        """
        Performs a forward pass and returns specified attributes.

        Parameters:
            to_return (list): List of attribute names to return.

        Returns:
            OrderedDict: Contains requested attributes after forward pass.
        """
        with torch.inference_mode():
            self.forward()
            return OrderedDict((name, getattr(self, name, None)) for name in to_return)

    def get_losses(self):
        """
        Retrieves current loss values for all components.

        Note:
            Ensure this is called after at least one optimization step.

        Returns:
            OrderedDict: Contains current loss values for key model components.
        """
        return OrderedDict(
            D_X=self.loss_D_X,
            D_Y=self.loss_D_Y,
            G=self.loss_G_adv_Y,
            F=self.loss_G_adv_X,
            cycle_X=self.loss_cycle_X,
            cycle_Y=self.loss_cycle_Y,
            identity_loss=self.identity_loss_G,
            total_loss = self.loss_G_total
        )

