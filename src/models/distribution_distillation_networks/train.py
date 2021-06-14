import torch
import numpy as np
from src.foolbox.adversarial_training import AttackModel


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


def compute_loss_accuracy(student_model, teacher_model, loader):
    with torch.no_grad():
        loss = 0
        for batch_index, (X, Y) in enumerate(loader):
            X = X.to(device)
            Y_pred = student_model(X, None, return_output='hard', compute_loss=False)
            if batch_index == 0:
                Y_pred_all = Y_pred.view(-1).to("cpu")
                Y_all = Y.view(-1).to("cpu")
            else:
                Y_pred_all = torch.cat([Y_pred_all, Y_pred.view(-1).to("cpu")], dim=0)
                Y_all = torch.cat([Y_all, Y.view(-1).to("cpu")], dim=0)

            for n, sub_model in enumerate(teacher_model.networks):
                soft_Y_pred = sub_model(X, None, return_output='soft', compute_loss=False)
                student_model(X, soft_Y_pred, compute_loss=True)
                loss += student_model.grad_loss.item()
        loss = loss / (Y_pred_all.size(0) * teacher_model.n_networks)
        accuracy = ((Y_pred_all == Y_all).float().sum() / Y_pred_all.size(0)).item()
    return loss, accuracy


def train(teacher_model, student_model, train_loader, val_loader, in_data_attack: AttackModel = None, rs_sigma=None,
          max_epochs=200, frequency=2, patience=5, model_path='saved_model', full_config_dict={}):
    print(device)
    teacher_model.to(device)
    student_model.to(device)
    print('\nStudent Model')
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_loss = float("Inf")
    for epoch in range(max_epochs):
        for batch_index, (X_train, Y_train) in enumerate(train_loader):
            X_train = X_train.to(device)
            for n, sub_model in enumerate(teacher_model.networks):
                with torch.no_grad():
                    soft_Y_pred = sub_model(X_train, None, return_output='soft', compute_loss=False)

                if rs_sigma is not None:
                    X_train = X_train + rs_sigma * torch.randn_like(X_train)

                if in_data_attack is not None:
                    student_model.eval()
                    adv_image, calibrated_labels = in_data_attack.attack(X_train, soft_Y_pred)
                    X_train = adv_image.detach()
                    # TODO: use calibrated_labels instead of labels

                student_model.train()
                student_model(X_train, soft_Y_pred, compute_loss=True)
                student_model.step()

        if epoch % frequency == 0:
            # Stats on data sets
            # train_loss, train_accuracy = student_compute_loss_accuracy(student_model, teacher_model, train_loader)
            # train_losses.append(round(train_loss, 3))
            # train_accuracies.append(round(train_accuracy, 3))

            val_loss, val_accuracy = compute_loss_accuracy(student_model, teacher_model, val_loader)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print("Epoch {} -> Val loss {} | Val Acc.: {}".format(epoch, round(val_losses[-1], 3), round(val_accuracies[-1], 3)))
            # print("Epoch ", epoch,
            #       "-> Train loss: ", train_losses[-1], "| Val loss: ", val_losses[-1], "| Test loss: ", test_losses[-1],
            #       "| Train Acc.: ", train_accuracies[-1], "| Val Acc.: ", val_accuracies[-1], "| Test Acc.: ", test_accuracies[-1])

            if best_val_loss > val_losses[-1]:
                best_val_loss = val_losses[-1]
                torch.save({'epoch': epoch, 'model_config_dict': full_config_dict, 'model_state_dict': student_model.state_dict(), 'loss': best_val_loss}, model_path)
                print('Model saved')

            if np.isnan(val_losses[-1]):
                print('Detected NaN Loss')
                break

            if int(epoch / frequency) > patience and val_losses[-patience] <= min(val_losses[-patience:]):
                print('Early Stopping.')
                break

    return train_losses, val_losses, train_accuracies, val_accuracies
