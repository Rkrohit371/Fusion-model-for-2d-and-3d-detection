# Fusion Architecture for 2D Image and 3D Point Cloud Data

This project implements a fusion neural network architecture that combines features extracted from 2D images and 3D point cloud data. The model predicts both the class of an object and the object's coordinates based on these combined features.

## Model Architecture

The model consists of the following components:

1. **CNN for 2D Images (`CNN_2D`)**:
   - A convolutional neural network (CNN) that processes 2D images to extract feature vectors.
   - The CNN includes 3 convolutional layers followed by fully connected layers.

2. **PointNet for 3D Point Clouds (`PointNet_3D`)**:
   - A neural network based on PointNet that processes 3D point cloud data.
   - It applies a series of fully connected layers followed by max-pooling to capture the global feature representation of the point cloud.

3. **Fusion Network (`FusionNet`)**:
   - The extracted features from both the CNN and PointNet are concatenated and passed through fully connected layers to generate two outputs:
     - **Class prediction**: Predicts the class label of the object.
     - **Coordinate prediction**: Predicts the 8 coordinates of the object.

## Prerequisites

To run this project, ensure you have the following dependencies installed:

- Python 3.7+
- PyTorch
- torchvision
- tqdm
- PIL (Python Imaging Library)
- NumPy

Install the required dependencies using the following command:

```bash
pip install torch torchvision tqdm Pillow numpy
```

## Dataset
The dataset consists of randomly generated 2D images and 3D point clouds:

- Images: Randomly generated using the save_random_images function. These are saved as .png files in the random_images folder.
- Point Clouds: Generated randomly with 1024 points and 3 dimensions (x, y, z coordinates).
- Labels: Random integer labels representing the class of the object.
Coordinates: Random 8-dimensional coordinate vectors representing the object's coordinates.

## Code Structure
- ImagePointCloudDataset: A custom PyTorch Dataset class that loads images, point clouds, labels, and coordinates.
- CNN_2D: A neural network for extracting features from 2D images.
- PointNet_3D: A neural network for extracting global features from 3D point clouds.
- FusionNet: The main fusion architecture that combines 2D and 3D features, and produces two outputs: class predictions and coordinates.
- FusionTrain: A class to handle the training process, including loss calculation and backpropagation.

## Running the Code
1. Generate random images:

    - Run the script with the function save_random_images to generate 1000 random images.
    - Point clouds, labels, and coordinates are also randomly generated.
    
    
2. Train the fusion model:
    - The training process includes minimizing two loss functions:
        - Cross-entropy loss for classification.
        - Mean squared error (MSE) loss for coordinate prediction.
    - The training logs include the total loss, class loss, and coordinate loss.

## Command to run the script:
```
python fusion_model.py
```

## Example Code
```
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
                               num_epochs=num_epochs,
                               dataloader=dataloader)

    training_obj.train()
```

## Training Details
During training:
    - The model is optimized using the Adam optimizer.
    - The losses are logged after every 10 batches to monitor training progress.

## Results
The model outputs two predictions:
1. Class output: The predicted class of the object.
2. Coordinate output: The predicted 8-dimensional coordinates of the object.

## Future Work
- Implement real-world datasets for training instead of random images and point clouds.
- Explore other fusion strategies, such as attention mechanisms for better feature merging.