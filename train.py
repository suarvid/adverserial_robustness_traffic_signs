import torch
from utils import print_info

VERY_RANDOM_SEED: int = 42  # To get same results each time

torch.manual_seed(VERY_RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(VERY_RANDOM_SEED)

# oklart om detta ska vara med
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def train_model(model, train_loader, optimizer, criterion, scheduler=None, epoch=None, device='cpu'):
    model.train()
    print_info("Training Model")
    train_running_loss = 0.0
    train_running_correct_classified = 0
    iter_counter = 0
    iters = len(train_loader)
    for i, data in enumerate(train_loader):
        iter_counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward Pass
        outputs = model(image)
        # Loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct_classified += (preds == labels).sum().item()
        # Backprop
        loss.backward()
        # Update weights
        optimizer.step()

        # Using a learning rate scheduler can yield higher training accuracy
        if scheduler is not None:  # What do if epoch is None?
            scheduler.step(epoch + i / iters)

    # Compute loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / iter_counter
    epoch_accuracy = 100.0 * (train_running_correct_classified / len(train_loader.dataset))

    return epoch_loss, epoch_accuracy


def validate_model(model, validation_loader, criterion, class_names, device='cpu'):
    model.eval()
    print_info("Evaluating Model")
    valid_running_loss = 0.0
    valid_running_correct_classified = 0
    iter_counter = 0

    # Two lists, to keep track of class-wise accuracy
    class_correct = list(0.0 for i in range(len(class_names)))
    class_total = list(0.0 for i in range(len(class_names)))

    with torch.no_grad():  # valdation
        for i, data in enumerate(validation_loader):
            iter_counter += 1

            image, labels = data
            image = image.to(device)
            labels = label.to(device)

            # Forward pass
            outputs = model(image)

            # Calculate loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            # Calculate accuracy
            _, predictions = torch.max(outputs.data, 1)  # TODO: Behövs 1:an här? Simplare att ta bort?
            valid_running_correct_classified += (predictions == labels).sum().item()

            # Calculate the accuracy for each class
            correct = (predictions == labels).squeeze()
            for i in range(len(predictions)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    epoch_loss = valid_running_loss / iter_counter
    epoch_acc = 100.0 * (valid_running_correct_classified / len(validation_loader.dataset))

    # Print class-wise accuracy after each epoch
    print()
    for i in range(len(class_names)):
        print(f"Accuracy of class {class_names[i]}: {100 * class_correct[i] / class_total[i]}")
    print()

    return epoch_loss, epoch_acc
