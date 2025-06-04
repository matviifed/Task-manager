import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

class LogoClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(LogoClassificationModel, self).__init__()
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class LogoDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, fashion_brands=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.brands = fashion_brands or []
        
        # Map brand names to class indices
        self.class_to_idx = {brand: idx for idx, brand in enumerate(self.brands)}
        
        # Load all images
        self.samples = self._load_dataset()
        
    def _load_dataset(self):
        samples = []
        images_path = os.path.join(self.root_dir, f"images/{self.split}")
        
        # Iterate through brand folders
        for brand in self.brands:
            brand_path = os.path.join(images_path, brand)
            if not os.path.exists(brand_path):
                print(f"Warning: Brand folder not found - {brand_path}")
                continue
                
            # Get all images for this brand
            for img_file in os.listdir(brand_path):
                if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                img_path = os.path.join(brand_path, img_file)
                class_id = self.class_to_idx[brand]
                samples.append((img_path, class_id))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert class_id to tensor
        class_id = torch.tensor(class_id)
        
        return image, class_id

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss and accuracy
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
    
    val_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return val_loss, accuracy

def split_dataset(dataset, val_ratio=0.2):
    """Split dataset into train and validation sets"""
    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    return torch.utils.data.random_split(dataset, [train_size, val_size])

def train_model(model, train_loader, val_loader, num_epochs=30, device='cuda'):
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_logo_classifier.pth')
            print(f"Saved new best model with validation accuracy: {val_acc:.2f}%")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print("-" * 50)
    
    return model, history

def predict_logo(model, image_path, transform, class_names, device='cuda'):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    
    # Transform image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        scores = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, class_idx = torch.max(scores, 0)
    
    class_name = class_names[class_idx.item()] if class_idx.item() < len(class_names) else "Unknown"
    
    return class_name, confidence.item()

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    # Define fashion brands
    fashion_brands = ['Adidas', 'Puma', 'NewBalance', 'prada', 'lacoste', 
                     'Gucci', 'louis vuitton', 'Balenciaga', 'Chrome Hearts', 
                     'Converse', 'Columbia', 'Hugo Boss', 'levis', 
                     'Ralph Lauren', 'versace', 'zara']
    
    # Set dataset path
    data_root = "C:\\Users\\Администратор\\OneDrive\\Робочий стіл\\LogoDet-3K\\Clothes"
    
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create full dataset and split into train/val
    full_dataset = LogoDataset(data_root, split='train', transform=None, fashion_brands=fashion_brands)
    train_dataset, val_dataset = split_dataset(full_dataset, val_ratio=0.2)
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Create model
    num_classes = len(fashion_brands)
    model = LogoClassificationModel(num_classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train model
    trained_model, history = train_model(model, train_loader, val_loader, num_epochs=30, device=device)
    
    # Plot training history
    plot_training_history(history)
    
    # Test prediction
    test_image_path = "path/to/your/test_image.jpg"  # Replace with your test image path
    if os.path.exists(test_image_path):
        class_name, confidence = predict_logo(
            trained_model, 
            test_image_path, 
            val_transform, 
            fashion_brands, 
            device
        )
        print(f"\nPrediction for test image:")
        print(f"Brand: {class_name}")
        print(f"Confidence: {confidence:.2%}")
    else:
        print("\nTest image not found. Please update the test_image_path variable.")