import torch
import os


def train(num_epochs, train_dataloader, test_dataloader, model, lr, device, early_stopping=3, save_weights=True):
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader), eta_min=0)

    steps_since_improvement = 0
    best_test_loss = float('inf')
    best_test_acc = 0

    for i in range(num_epochs):
        train_loss, train_acc = train_model(model, optimizer, train_dataloader, device)
        test_loss, test_acc = test_model(model, test_dataloader, device)
        scheduler.step()
        # Update best model and steps since improvement
        if best_test_acc < test_acc:
            steps_since_improvement = 0
            best_test_loss = test_loss
            best_test_acc = test_acc
            if save_weights:
                save_model(model, path='./trained_models/', file_name='model.ptr')
        else:
            steps_since_improvement += 1

        # Stop training early if early stopping steps have been reached
        if early_stopping is not None and early_stopping < steps_since_improvement:
            print('Stopping training early')
            break

        # Display Epoch performance
        print(f'Epoch: {i+1} | Train Loss: {train_loss} | Train Acc: {train_acc} | Test Loss: {test_loss} | Test Acc: {test_acc}')

    # Save final model
    if save_weights:
        print('Saving model...')
        save_model(model, path='./trained_models/', file_name='model.ptr')
    print(f'<Final Test Performance> Test Loss: {best_test_loss} | Test Acc: {best_test_acc}')

    # Return best metric
    return best_test_acc

def train_model(model, optimizer, train_dataloader, device):
    total_correct = 0
    total_loss = 0

    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        loss, num_correct = model.calc_loss(inputs, targets)

        total_loss += loss.item()
        total_correct += num_correct

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(train_dataloader), total_correct / len(train_dataloader.dataset)

@torch.no_grad()
def test_model(model, test_dataloader, device):
    model.eval()

    total_correct = 0
    total_loss = 0

    for inputs, targets in test_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        loss, num_correct = model.calc_loss(inputs, targets)

        total_loss += loss.item()
        total_correct += num_correct

    return total_loss / len(test_dataloader), total_correct / len(test_dataloader.dataset)

def save_model(model, path, file_name):
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, file_name)
    torch.save(model.state_dict(), full_path)
