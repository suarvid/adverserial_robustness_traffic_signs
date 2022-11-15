import torch.cuda
import torch.optim as optim
import torch.nn as nn

from datasets import get_datasets, get_data_loaders
from utils import print_info, save_model, save_plots
import argparse

from model import build_model
from train import train_model, validate_model

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    '-e', '--epochs', type=int, default=10, help="Number of epochs to train the network for"
)

arg_parser.add_argument(
    '-lr', '--learning-rate', type=float, dest='learning-rate', default=0.001,
    help="Learning rate for training the model"
)

arg_parser.add_argument(
    '-pt', '--pretrained', action='store_true', help="Sets whether to use pretrained weights or not"
)

arg_parser.add_argument(
    '-ft', '--fine-tune', dest='fine-tune', action='store_true', help="Sets whether to train all layers or not"
)


args = vars(arg_parser.parse_args())

if __name__ == '__main__':
    dataset_train, dataset_valid, dataset_classes = get_datasets()
    print_info(f"Number of training images: {len(dataset_train)}")
    print_info(f"Number of validation images: {len(dataset_valid)}")
    print_info(f"Class names: {dataset_classes}\n")

    train_loader, validation_loader = get_data_loaders(dataset_train, dataset_valid)

    # Get parameters from cmd-line arguments
    learning_rate = args['learning-rate']
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs to train for: {epochs}")

    model = build_model(
        pretrained=args['pretrained'],
        fine_tune=args['fine-tune'],
        num_classes=len(dataset_classes)
    ).to(device)

    num_total_params = sum(param.numel() for param in model.parameters())
    num_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Total Parameters in Model: {num_total_params}")
    print(f"Trainable Parameters in Model: {num_trainable_params}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=1,
        verbose=True
    )

    # Keep track of losses and accuracies
    training_loss, validation_loss = [], []
    training_accuracy, vaildation_accuracy = [], []

    # Start training
    for epoch in range(epochs):
        print_info(f"Epoch {epoch} of {epochs}")
        train_epoch_loss, train_epoch_acc = train_model(model, train_loader, optimizer, criterion, scheduler, epoch)
        valid_epoch_loss, valid_epoch_acc = validate_model(model, validation_loader, criterion, dataset_classes)

        training_loss.append(train_epoch_loss)
        validation_loss.append(valid_epoch_loss)
        training_accuracy.append(train_epoch_acc)
        vaildation_accuracy.append(valid_epoch_acc)

        print(f"Training Loss: {train_epoch_loss:.3f}, Training Accuracy: {train_epoch_acc:.3f}")
        print(f"Validation Loss: {valid_epoch_loss:.3f}, Validation Accuracy: {vaildation_accuracy:.3f}")
        print('-'*50)

    save_model(epochs, model, optimizer, criterion)
    save_plots(training_accuracy, vaildation_accuracy, training_loss, validation_loss)
    print("Training complete")
