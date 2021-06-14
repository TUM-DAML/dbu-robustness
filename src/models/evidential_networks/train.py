import torch
import numpy as np
from src.foolbox.adversarial_training import AttackModel


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


def compute_loss_accuracy(model, loader, epoch):
    model.eval()
    with torch.no_grad():
        loss = 0
        for batch_index, (X, Y) in enumerate(loader):
            X, Y = X.to(device), Y.to(device)
            Y_pred = model(X, Y, return_output='hard', compute_loss=True, epoch=10.)
            if batch_index == 0:
                Y_pred_all = Y_pred.view(-1).to("cpu")
                Y_all = Y.view(-1).to("cpu")
            else:
                Y_pred_all = torch.cat([Y_pred_all, Y_pred.view(-1).to("cpu")], dim=0)
                Y_all = torch.cat([Y_all, Y.view(-1).to("cpu")], dim=0)
            loss += model.grad_loss.item()
        loss = loss / Y_pred_all.size(0)
        accuracy = ((Y_pred_all == Y_all).float().sum() / Y_pred_all.size(0)).item()
    model.train()
    return loss, accuracy


def train(model, train_loader, val_loader, in_data_attack: AttackModel = None, rs_sigma=None,
          max_epochs=200, frequency=2, patience=5, model_path='saved_model', full_config_dict={}):
    model.to(device)
    model.train()
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_loss = float("Inf")

    for epoch in range(max_epochs):
        for batch_index, (X_train, Y_train) in enumerate(train_loader):
            X_train, Y_train = X_train.to(device), Y_train.to(device)

            if rs_sigma is not None:
                X_train = X_train + rs_sigma * torch.randn_like(X_train)

            if in_data_attack is not None:
                model.eval()
                adv_image, calibrated_labels = in_data_attack.attack(X_train, Y_train)
                X_train = adv_image.detach()
                Y_train = calibrated_labels.detach()

            model.train()
            model(X_train, Y_train, compute_loss=True, epoch=epoch)
            model.step()

        if epoch % frequency == 0:
            # Stats on data sets
            # train_loss, train_accuracy = compute_loss_accuracy(model, train_loader)
            # train_losses.append(round(train_loss, 3))
            # train_accuracies.append(round(train_accuracy, 3))

            val_loss, val_accuracy = compute_loss_accuracy(model, val_loader, epoch)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print("Epoch {} -> Val loss {} | Val Acc.: {}".format(epoch, round(val_losses[-1], 3), round(val_accuracies[-1], 3)))
            # print("Epoch ", epoch,
            #       "-> Train loss: ", train_losses[-1], "| Val loss: ", val_losses[-1],
            #       "| Train Acc.: ", train_accuracies[-1], "| Val Acc.: ", val_accuracies[-1])

            if best_val_loss > val_losses[-1]:
                best_val_loss = val_losses[-1]
                torch.save({'epoch': epoch, 'model_config_dict': full_config_dict, 'model_state_dict': model.state_dict(), 'loss': best_val_loss}, model_path)
                print('Model saved')

            if np.isnan(val_losses[-1]):
                print('Detected NaN Loss')
                break

            if int(epoch / frequency) > patience and val_losses[-patience] <= min(val_losses[-patience:]):
                print('Early Stopping.')
                break

    return train_losses, val_losses, train_accuracies, val_accuracies
