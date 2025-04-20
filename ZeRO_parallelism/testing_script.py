import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# ----------- Settings -----------
data_path = "/home/mariam.barakat/Downloads/ml710project/imagenet_10class"  # replace with your actual path
model_path = "/home/mariam.barakat/Downloads/ml710project/Mariam/output/Zero_3/vgg16_10_Zero_3.pth"  # replace with your actual .pth file path
batch_size = 32
use_fp16 = False  # set to True if you trained with fp16

# ----------- Load Test Set -----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_path = os.path.join(data_path, 'test')
testset = torchvision.datasets.ImageFolder(test_path, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# ----------- Load Model -----------
model = torchvision.models.vgg16()
model.classifier[6] = torch.nn.Linear(4096, 2)  # assuming 2 classes
model.load_state_dict(torch.load(model_path))
model.eval()
model.cuda()

if use_fp16:
    model.half()

# ----------- Evaluate Model -----------
correct = 0
total = 0
criterion = torch.nn.CrossEntropyLoss()
total_loss = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.cuda(), labels.cuda()
        if use_fp16:
            images = images.half()

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
avg_loss = total_loss / len(testloader)

print(f"\n✅ Test Accuracy: {accuracy:.2f}%")
print(f"📉 Test Loss: {avg_loss:.4f}")
