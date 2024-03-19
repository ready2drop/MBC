import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import StepLR

from dataset.dataloader import getloader
from model.mnn import MultiModalClassifier

from timeit import default_timer as timer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} ')

data_dir = 'RawData/jpeg-melanoma-512x512/'
# Initialize the model
num_classes = 1  # Assuming binary classification
num_gpus = 8
num_features = 1280  # Number of features extracted by EfficientNet-B0
batch_size = 200


train_loader, valid_loader = getloader(data_dir, batch_size=batch_size, mode='train')

    
model = MultiModalClassifier(num_classes, num_features)
model = DataParallel(model, device_ids=[i for i in range(num_gpus)]).to(device)
print("Model loaded & Paralleled successfully.")

# Start time of training loop
start_training_time = timer()

# Define loss function (Binary Cross-Entropy) and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.004)

scheduler = StepLR(optimizer, step_size=1, gamma=0.1)  # Reduce learning rate by a factor of 0.1 every 1 epochs

# Training loop
num_epochs = 10
best_val_acc = 0.0
best_epoch = 1

train_losses, val_losses, train_accs, val_accs = [],[],[],[]

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, ages, anatom_sites, sexes, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        outputs = model(images.to(device), ages.to(device), anatom_sites.to(device), sexes.to(device))
        loss = criterion(outputs.squeeze(), labels.float().to(device))  # Squeeze output and convert labels to float
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0).squeeze().long()  # Convert outputs to binary predictions
        total_train += labels.size(0)
        correct_train += (predicted == labels.to(device)).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation
    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, ages, anatom_sites, sexes, labels in tqdm(valid_loader, desc=f"Validation"):
            outputs = model(images.to(device), ages.to(device), anatom_sites.to(device), sexes.to(device))
            loss = criterion(outputs.squeeze(), labels.float().to(device))  # Squeeze output and convert labels to float
            val_running_loss += loss.item()
            predicted = (outputs > 0).squeeze().long()  # Convert outputs to binary predictions
            total_val += labels.size(0)
            correct_val += (predicted == labels.to(device)).sum().item()

    val_loss = val_running_loss / len(valid_loader)
    val_acc = correct_val / total_val
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
    
    # Step the scheduler
    scheduler.step()
    
    # Save the best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f'best_epoch_weights.pth')
        best_epoch = epoch+1
        
print(f"The Best epoch is {best_epoch}")

# End time of training loop
end_training_time = timer()

# Calculate total training time
total_training_time = (end_training_time - start_training_time) / 3600
print(f"Total Training Time: {total_training_time:.2f} hours")