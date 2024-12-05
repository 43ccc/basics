import torch
import os
import torch.nn as nn

def train(num_epochs, train_dataloader, test_dataloader, model, lr, device):
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader), eta_min=0)

    for i in range(num_epochs):
        train_loss, train_acc = train_model(model, optimizer, train_dataloader, device)
        test_loss, test_acc = test_model(model, test_dataloader, device)
        scheduler.step()

        # Display Epoch performance
        print(f'Epoch: {i+1} | Train Loss: {train_loss} | Train Acc: {train_acc} | Test Loss: {test_loss} | Test Acc: {test_acc}')

    # Save final model
    save_model(model, path='./trained_models/', file_name='finetuned_model.ptr')

def train_model(model, optimizer, train_dataloader, device):
    model.train()
    total_correct = 0
    total_loss = 0
    loss_function = nn.CrossEntropyLoss()

    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        pred = model(inputs)
        loss = loss_function(pred, targets)
        num_correct = sum(pred.argmax(dim=1) == targets)

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
    loss_function = nn.CrossEntropyLoss()

    for inputs, targets in test_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        pred = model(inputs)
        loss = loss_function(pred, targets)
        num_correct = sum(pred.argmax(dim=1) == targets)

        total_loss += loss.item()
        total_correct += num_correct

    return total_loss / len(test_dataloader), total_correct / len(test_dataloader.dataset)

def save_model(model, path, file_name):
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, file_name)
    torch.save(model.state_dict(), full_path)
