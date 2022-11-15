import torch
import matplotlib.pyplot as plt



def print_info(msg: str):
    print(f'[INFO] {msg}')


def save_model(epochs, model, optimizer, criterion):
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, f"../output/model.pth")


def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    # Accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', label='training accuracy'
    )
    plt.plt(valid_acc, color='blue', linestyle='-', label='validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"../output/accuracy.png") # oklart varför dessa ska vara format-strings?

    # Loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', label='train loss'
    )
    plt.plot(valid_loss, color='red', linestyle='-', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"../output/loss.png")

