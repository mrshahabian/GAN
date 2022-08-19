import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST  # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from generator import get_generator_block, Generator, get_noise
from discriminator import get_discriminator_block, Discriminator
from test import test_gen_block, test_generator, test_get_noise
from test import test_disc_block, test_discriminator
from test import test_disc_reasonable, test_disc_loss, test_gen_loss, test_gen_reasonable
torch.manual_seed(0)  # Set for testing purposes, please do not change!


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()



# Verify the generator block function
test_gen_block(25, 12)
test_gen_block(15, 28)
print(" Generator Block Success!")

# Verify the generator class
test_generator(5, 10, 20)
test_generator(20, 8, 24)
print("Generator class Success!")

# Verify the noise vector function
test_get_noise(1000, 100, 'cpu')
if torch.cuda.is_available():
    test_get_noise(1000, 32, 'cuda')
print("Noise Success!")

# Verify the discriminator block function
test_disc_block(25, 12)
test_disc_block(15, 28)
print("Discriminator Success!")

# Verify the discriminator class
test_discriminator(5, 10)
test_discriminator(20, 8)
print("Discriminator class Success!")

test_disc_reasonable()
test_disc_loss()
print("train discriminator Success!")

test_gen_reasonable(10)
test_gen_loss(18)
print("train generator Success!")