import os
import tqdm
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from generate_images import save_random_images

class ImagePointCloudDataset(Dataset):
    def __init__(self, image_paths, point_clouds, labels, coordinates, transform=None):
        self.image_paths = image_paths
        self.point_clouds = point_clouds
        self.labels = labels
        self.coordinates = coordinates
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        point_cloud = self.point_clouds[idx]
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        coord = torch.tensor(self.coordinates[idx], dtype=torch.float32)

        return image, point_cloud, label, coord


class CNN_2D(nn.Module):
    def __init__(self):
        super(CNN_2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class PointNet_3D(nn.Module):
    def __init__(self):
        super(PointNet_3D, self).__init__()
        self.fc1 = nn.Linear(3, 64)    
        self.fc2 = nn.Linear(64, 128)  
        self.fc3 = nn.Linear(128, 1024)  

        self.global_fc1 = nn.Linear(1024, 512)  
        self.global_fc2 = nn.Linear(512, 256)   
        
    def forward(self, x):
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x)) 
        x = F.relu(self.fc3(x))
        x = torch.max(x, 1, keepdim=False)[0]  
        x = F.relu(self.global_fc1(x))  
        x = self.global_fc2(x) 
        
        return x

class FusionNet(nn.Module):
    def __init__(self, num_classes=10):
        super(FusionNet, self).__init__()
        self.cnn_2d = CNN_2D()
        self.pointnet_3d = PointNet_3D()
        
        self.fc_fusion = nn.Linear(128 + 256, 512)  # 128 from CNN_2D, 256 from PointNet_3D
        
        self.fc_class = nn.Linear(512, num_classes)
        self.fc_coordinates = nn.Linear(512, 8)
        
    def forward(self, img, point_cloud):
        features_2d = self.cnn_2d(img)
        features_3d = self.pointnet_3d(point_cloud)
        
        fused_features = torch.cat((features_2d, features_3d), dim=1)
        
        fused_features = F.relu(self.fc_fusion(fused_features))
        
        class_output = self.fc_class(fused_features)
        coordinates_output = self.fc_coordinates(fused_features)
        
        return class_output, coordinates_output

class FusionTrain:
    def __init__(self, num_classes, num_epochs, dataloader):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.model = FusionNet(num_classes=self.num_classes).to(self.device)
        self.criterion_class = nn.CrossEntropyLoss()  
        self.criterion_coord = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def train(self):
        self.model.train()

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            running_class_loss = 0.0
            running_coord_loss = 0.0

            for i, (images, point_clouds, labels, coordinates) in enumerate(tqdm.tqdm(dataloader)):
                images = images.to(self.device)
                point_clouds = point_clouds.to(self.device)
                labels = labels.to(self.device)
                coordinates = coordinates.to(self.device)

                self.optimizer.zero_grad()

                class_outputs, coord_outputs = self.model(images, point_clouds)

                loss_class = self.criterion_class(class_outputs, labels)
                loss_coord = self.criterion_coord(coord_outputs, coordinates)
                loss = loss_class + loss_coord

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                running_class_loss += loss_class.item()
                running_coord_loss += loss_coord.item()
                if i % 10 == 9:
                    print(f"Epoch {epoch+1}, Batch {i+1}, Total Loss: {running_loss/10:.4f}, "
                        f"Class Loss: {running_class_loss/10:.4f}, Coord Loss: {running_coord_loss/10:.4f}")
                    running_loss = 0.0
                    running_class_loss = 0.0
                    running_coord_loss = 0.0


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  
        transforms.ToTensor(),        
    ])

    num_images = 1000
    num_classes = 5
    num_epochs = 100

    save_random_images(num_images=num_images, image_size=(64, 64))
    point_clouds = [np.random.randn(1024, 3) for _ in range(num_images)]

    image_paths = []
    folder_path = "random_images"
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        image_paths.append(filepath)


    labels = np.random.randint(0, num_classes, size=num_images)
    coordinates = np.random.randn(num_images, 8)  

    dataset = ImagePointCloudDataset(image_paths=image_paths, point_clouds=point_clouds, labels=labels, coordinates=coordinates, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    training_obj = FusionTrain(num_classes=num_classes,
                               num_epochs=num_classes,
                               dataloader=dataloader)
    
    training_obj.train()
