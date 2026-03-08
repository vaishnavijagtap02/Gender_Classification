import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

def train_model():

    BATCH_SIZE = 32
    EPOCHS = 5   # Increased from 3 (better learning)
    TEAM_FOLDER = "Team_40_InnovHERS_GenderClassification"

    # Data Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    print("1️⃣ Loading Data...")

    train_path = os.path.join('dataset', 'Training')

    if not os.path.exists(train_path):
        print("❌ dataset/Training folder not found!")
        return

    train_data = datasets.ImageFolder(train_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print("📌 Class mapping:", train_data.class_to_idx)

    # Build Model
    print("2️⃣ Building MobileNetV3...")

    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, 2
    )

    device = torch.device("cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    print("3️⃣ Training Started...")

    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {i} | Loss: {loss.item():.4f}")

        print(f"✅ Epoch {epoch+1} Finished | Avg Loss: {running_loss/len(train_loader):.4f}")

    # Save Model
    print("4️⃣ Saving Model...")

    save_path = os.path.join(TEAM_FOLDER, "model")
    os.makedirs(save_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))

    print("🎉 Training Complete! Model saved successfully.")

if __name__ == "__main__":
    train_model()