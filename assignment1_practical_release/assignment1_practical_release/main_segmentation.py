"""
This is the main script for training and evaluating the Unet model on the segmentation of retina blood vessels.

Input:
    - dataset: path to the dataset (the Data folder containing train and test folders)
    - logdir: str, directory to save logs
    - batch_size: int, batch size
    - num_epochs: int, number of epochs
    - lr: float, learning rate
    - weight_decay: float, weight decay
    - print_every: int, print every

Author:
    Credits to the work done previously by Johan Obando and Jerry Huang
    Pierre-Louis Benveniste
"""
import argparse
import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from unet import UNet
from utils import DiceCELoss, DiceLoss
from glob import glob
import cv2
from torch.utils.data import Dataset
import json
import random

# seed experiment
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.benchmark = True


class GetDataset(Dataset):
    def __init__(self, images_path, masks_path,augmentation=None):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.augmentation = augmentation

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image/255.0


        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask/255.0

        if self.augmentation:
            image, mask = self.augmentation(image, mask)

        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples


def parse_args():
    """
    This function parses the command line arguments.
    """
    parser = argparse.ArgumentParser(description='Retina blood vessel segmentation training script')
    parser.add_argument('--dataset', type=str, default=None, help='Path to the dataset (the Data folder containing train and test folders)', required=True)
    parser.add_argument('--logdir', type=str, default=None, help='Directory to save logs', required=True)
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (default: %(default)s).')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs for training (default: %(default)s).')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: %(default)s).')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default: %(default)s).')
    parser.add_argument('--print_every', type=int, default=80, help='Number of minibatches after which we print the loss (default: %(default)s).')
    return parser.parse_args()


def dice_score(preds, targets):
    # Compute the dice score using the DiceLoss class
    dice_loss = DiceLoss()
    loss = dice_loss(preds, targets)
    dice = 1-loss
    return dice


def train(epoch, model, dataloader, optimizer, loss_fn, accuracy_fn, device, args):
    model.train()
    total_iters = 0
    epoch_accuracy=0
    epoch_loss=0
    start_time = time.time()
    for idx, (X,y) in enumerate(dataloader):
        # Format the data
        y = y.squeeze().long()
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)
 
        # Compute loss and accuracy
        batch_loss = loss_fn(y_pred, y)
        loss = batch_loss/args.batch_size
        acc = accuracy_fn(y_pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update epoch accuracy and loss
        epoch_accuracy += acc.item() / len(dataloader)
        epoch_loss += loss.item() / len(dataloader)
        total_iters += 1

        # Print every args.print_every iterations
        if idx % args.print_every == 0:
            tqdm.write(f"[TRAIN] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}")
    tqdm.write(f"== [TRAIN] Epoch: {epoch}, Accuracy: {epoch_accuracy:.3f} ==>")
    return epoch_loss, epoch_accuracy, time.time() - start_time


def evaluate(epoch, model, dataloader, loss_fn, accuracy_fn, device, args, mode="val"):
    model.eval()
    epoch_accuracy=0
    epoch_loss=0
    total_iters = 0
    start_time = time.time()
    with torch.no_grad():
        for idx, (X,y) in enumerate(dataloader):
            # Format the data
            y = y.squeeze().long()
            X, y = X.to(device), y.to(device)

            # Forward pass
            logits = model(X)
            
            # Compute loss and accuracy
            batch_loss = loss_fn(logits, y)
            loss = batch_loss/args.batch_size
            acc = accuracy_fn(logits, y)

            # Update epoch accuracy and loss
            epoch_accuracy += acc.item() / len(dataloader)
            epoch_loss += loss.item() / len(dataloader)
            total_iters += 1

            # Print every args.print_every iterations
            if idx % args.print_every == 0:
                tqdm.write(
                    f"[{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}"
                )
        tqdm.write(
            f"=== [{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Accuracy: {epoch_accuracy:.3f} ===>"
        )
    return epoch_loss, epoch_accuracy, time.time() - start_time


def main():
    # Parse the arguments
    args = parse_args()

    # Build output folder
    os.makedirs(args.logdir, exist_ok=True)

    # Check for the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # We load the dataset
    dataset_path = args.dataset
    train_x = sorted(glob(f"{dataset_path}/train/image/*"))
    train_y = sorted(glob(f"{dataset_path}/train/mask/*"))
    valid_x = sorted(glob(f"{dataset_path}/test/image/*"))
    valid_y = sorted(glob(f"{dataset_path}/test/mask/*"))
    print(f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n")
    # Load the datasets with the custom Dataset class
    train_set = GetDataset(train_x, train_y)
    val_set = GetDataset(valid_x, valid_y)
    
    train_set_no_aug = GetDataset(train_x, train_y)

    # take same index
    img_orig, mask_orig = train_set_no_aug[0]
    img_aug, mask_aug = train_set[0]

    # visualize
    visualize_before_after(img_orig, mask_orig, img_aug, mask_aug)
    # Load model
    print(f'Build UNET model...')
    model = UNet(input_shape=3, num_classes=1)
    model.to(device)
    print(f"Initialized UNET model with {sum(p.numel() for p in model.parameters())} "
          f"total parameters, of which {sum(p.numel() for p in model.parameters() if p.requires_grad)} are learnable.")
    print("Model architecture:\n", model)
    print("\n")
    
    # Optimizer
    optimizer = optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Loss function
    loss_fn = DiceCELoss()
    # Accuracy function
    accuracy_fn = dice_score
    
    # Initialize lists to store metrics
    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []
    train_times, valid_times = [], []
    
    # We define a set of data loaders that we can use for various purposes later.
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    valid_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)

    # Train and evaluate the model
    print(f'Training UNET model...')
    for epoch in tqdm(range(args.epochs)):
        tqdm.write(f"====== Epoch {epoch} ======>")

        # Train the model
        loss, acc, wall_time = train(epoch, model, train_dataloader, optimizer, loss_fn, accuracy_fn, device, args)
        train_losses.append(loss)
        train_accs.append(acc)
        train_times.append(wall_time)

        # Evaluate the model
        loss, acc, wall_time = evaluate(epoch, model, valid_dataloader, loss_fn, accuracy_fn, device, args)
        valid_losses.append(loss)
        valid_accs.append(acc)
        valid_times.append(wall_time)

        # Save log if logdir provided
    if args.logdir is not None:
        print(f'Writing training logs to {args.logdir}...')
        os.makedirs(args.logdir, exist_ok=True)
        with open(os.path.join(args.logdir, 'results.json'), 'w') as f:
            f.write(json.dumps(
                {
                    "train_losses": train_losses,
                    "valid_losses": valid_losses,
                    "train_accs": train_accs,
                    "valid_accs": valid_accs,
                },
                indent=4,
            ))

def geometric_augmentation(image, mask):
    h, w, _ = image.shape

    # Rotation légère
    angle = random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h))
    mask = cv2.warpAffine(mask, M, (w, h))

    # Flip horizontal
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def photometric_augmentation(image, mask):
    # Brightness
    factor = random.uniform(0.8, 1.2)
    image = np.clip(image * factor, 0, 1)

    # Contrast
    mean = np.mean(image, axis=(0,1), keepdims=True)
    contrast_factor = random.uniform(0.8, 1.2)
    image = np.clip((image - mean) * contrast_factor + mean, 0, 1)

    return image, mask


def noise_blur_augmentation(image, mask):
    # Gaussian blur
    if random.random() > 0.5:
        image = cv2.GaussianBlur(image, (5,5), 0)

    # Gaussian noise
    noise = np.random.normal(0, 0.02, image.shape)
    image = np.clip(image + noise, 0, 1)

    return image, mask


def combined_augmentation(image, mask):
    image, mask = geometric_augmentation(image, mask)
    image, mask = photometric_augmentation(image, mask)
    return image, mask


import matplotlib.pyplot as plt

def visualize_before_after(img_orig, mask_orig, img_aug, mask_aug):
    img_orig = img_orig.permute(1, 2, 0).cpu().numpy()
    mask_orig = mask_orig.squeeze().cpu().numpy()

    img_aug = img_aug.permute(1, 2, 0).cpu().numpy()
    mask_aug = mask_aug.squeeze().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    axes[0,0].imshow(img_orig)
    axes[0,0].set_title("Original Image")
    axes[0,1].imshow(mask_orig, cmap="gray")
    axes[0,1].set_title("Original Mask")

    axes[1,0].imshow(img_aug)
    axes[1,0].set_title("Augmented Image")
    axes[1,1].imshow(mask_aug, cmap="gray")
    axes[1,1].set_title("Augmented Mask")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()