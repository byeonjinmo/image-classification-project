import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import matplotlib.cm as cm
from sklearn.manifold import TSNE

class ImageDataset(Dataset):
    """Custom Dataset for loading images from a folder."""
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(folder_path) 
                              if os.path.isdir(os.path.join(folder_path, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(folder_path, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class ImageClassifier:
    """Framework for automating image classification with ResNet."""
    def __init__(self, data_dir, output_dir="results", batch_size=32, num_epochs=10, 
                 lr=0.001, model_type="resnet18", device=None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.model_type = model_type
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up preprocessing transformations
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Store the intermediate results
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
        
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_names = None
        self.features = []
        self.true_labels = []
    
    def load_data(self, val_split=0.2):
        """Load and split the data into training and validation sets."""
        print("Loading and preprocessing data...")
        dataset = ImageDataset(self.data_dir, transform=self.train_transform)
        self.class_names = dataset.classes
        print(f"Found {len(dataset)} images in {len(self.class_names)} classes")
        
        # Calculate split sizes
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        
        # Perform the split
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create train and validation datasets with appropriate transforms
        self.train_dataset = ImageDataset(self.data_dir, transform=self.train_transform)
        self.val_dataset = ImageDataset(self.data_dir, transform=self.val_transform)
        
        # Update the samples based on the split
        self.train_dataset.samples = [dataset.samples[i] for i in train_dataset.indices]
        self.val_dataset.samples = [dataset.samples[i] for i in val_dataset.indices]
        
        print(f"Training on {len(self.train_dataset)} images, validating on {len(self.val_dataset)} images")
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
        
        # Visualize sample images
        self._visualize_samples()
        
        return self.train_loader, self.val_loader
    
    def _visualize_samples(self, num_samples=5):
        """Visualize sample images from each class."""
        print("Visualizing sample images...")
        
        fig, axes = plt.subplots(len(self.class_names), num_samples, 
                                figsize=(num_samples*2, len(self.class_names)*2))
        
        # Inverse normalization function
        inv_normalize = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
        ])
        
        for class_idx, class_name in enumerate(self.class_names):
            # Get all images from this class
            class_images = [img_path for img_path, label in self.train_dataset.samples 
                           if label == self.train_dataset.class_to_idx[class_name]]
            
            # Select random samples
            if len(class_images) >= num_samples:
                sample_paths = np.random.choice(class_images, num_samples, replace=False)
            else:
                sample_paths = class_images
            
            # Display images
            for j, img_path in enumerate(sample_paths):
                if j < num_samples:
                    img = Image.open(img_path).convert('RGB')
                    img = self.val_transform(img)
                    img = inv_normalize(img)
                    img = img.permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1)
                    
                    if len(self.class_names) > 1:
                        ax = axes[class_idx, j]
                    else:
                        ax = axes[j]
                    
                    ax.imshow(img)
                    ax.set_title(class_name)
                    ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sample_images.png'))
        plt.close()
        print(f"Sample images saved to {os.path.join(self.output_dir, 'sample_images.png')}")
    
    def create_model(self):
        """Create and configure the ResNet model."""
        print(f"Creating {self.model_type} model...")
        
        # Select model architecture
        if self.model_type == "resnet18":
            model = models.resnet18(weights="IMAGENET1K_V1")
        elif self.model_type == "resnet34":
            model = models.resnet34(weights="IMAGENET1K_V1")
        elif self.model_type == "resnet50":
            model = models.resnet50(weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Replace the final fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(self.class_names))
        
        # Add a hook to get intermediate features
        def hook_fn(module, input, output):
            self.features.append(output.cpu().detach())
        
        model.avgpool.register_forward_hook(hook_fn)
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )
        
        return self.model
    
    def train(self, start_epoch=0):
        """Train the model and track progress.
        Args:
            start_epoch (int): 시작할 에폭 번호 (체크포인트에서 이어서 학습할 때 사용)
        """
        print(f"Starting training from epoch {start_epoch+1} for {self.num_epochs} epochs...")
        
        # 체크포인트에서 불러온 학습 히스토리가 있는지 확인
        if start_epoch > 0 and not self.history['train_loss']:
            print("경고: 체크포인트에서 학습 히스토리를 불러오지 못했습니다. 새로운 히스토리를 생성합니다.")
        
        for epoch in range(start_epoch, self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
            for inputs, labels in train_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Track statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                train_bar.set_postfix(
                    loss=f"{train_loss/train_total:.4f}", 
                    acc=f"{100.0*train_correct/train_total:.2f}%"
                )
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")
                for inputs, labels in val_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    # Track statistics
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # Update progress bar
                    val_bar.set_postfix(
                        loss=f"{val_loss/val_total:.4f}", 
                        acc=f"{100.0*val_correct/val_total:.2f}%"
                    )
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
            # Store metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Adjust learning rate
            self.scheduler.step(val_loss)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.num_epochs} summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'history': self.history
            }, os.path.join(self.output_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        
        # After training, visualize the results
        self._visualize_training_progress()
        
        print("Training completed!")
        return self.history
    
    def _visualize_training_progress(self):
        """Plot training and validation metrics."""
        print("Generating training progress visualization...")
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Create a figure with two subplots
        plt.figure(figsize=(12, 5))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        plt.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_progress.png'))
        plt.close()
        print(f"Training progress visualization saved to {os.path.join(self.output_dir, 'training_progress.png')}")
    
    def evaluate(self):
        """Evaluate the model on the validation set and generate visualizations."""
        print("Evaluating model and generating visualizations...")
        
        if not self.model:
            raise ValueError("Model not created or trained yet")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        self.features = []
        self.true_labels = []
        
        # Collect predictions and features
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Evaluating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                self.true_labels.extend(labels.cpu().numpy())
        
        # Generate confusion matrix
        self._plot_confusion_matrix(all_labels, all_preds)
        
        # Generate classification report
        report = classification_report(
            all_labels, all_preds, 
            target_names=self.class_names, 
            output_dict=True
        )
        
        # Save classification report as CSV
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(self.output_dir, 'classification_report.csv'))
        
        # Plot t-SNE visualization of features
        self._plot_tsne_visualization()
        
        # Generate class activation maps for sample images
        self._generate_cam_visualizations()
        
        print("Evaluation and visualization completed!")
        return report
    
    def _plot_confusion_matrix(self, true_labels, pred_labels):
        """Plot and save confusion matrix."""
        print("Generating confusion matrix...")
        
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()
        print(f"Confusion matrix saved to {os.path.join(self.output_dir, 'confusion_matrix.png')}")
    
    def _plot_tsne_visualization(self):
        """Generate t-SNE visualization of features."""
        print("Generating t-SNE visualization...")
        
        # Stack all features
        features = torch.cat(self.features).numpy()
        features = features.reshape(features.shape[0], -1)
        
        # Apply t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features)
        
        # Plot t-SNE
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(self.class_names):
            mask = np.array(self.true_labels) == i
            plt.scatter(
                features_tsne[mask, 0], features_tsne[mask, 1],
                label=class_name, alpha=0.7
            )
        
        plt.title('t-SNE Visualization of Features')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'tsne_visualization.png'))
        plt.close()
        print(f"t-SNE visualization saved to {os.path.join(self.output_dir, 'tsne_visualization.png')}")
    
    def _generate_cam_visualizations(self, num_samples=5):
        """Generate Class Activation Map visualizations."""
        print("Generating Class Activation Map visualizations...")
        
        # Get the weight matrix from the final fully connected layer
        fc_weights = self.model.fc.weight.data.cpu().numpy()
        
        # Sample a few images from each class for visualization
        fig, axes = plt.subplots(len(self.class_names), num_samples, 
                                figsize=(num_samples*3, len(self.class_names)*3))
        
        for class_idx, class_name in enumerate(self.class_names):
            # Get all images from this class in validation set
            class_samples = [(i, img_path) for i, (img_path, label) in enumerate(self.val_dataset.samples) 
                            if label == self.val_dataset.class_to_idx[class_name]]
            
            # Select random samples
            if len(class_samples) >= num_samples:
                samples = np.random.choice(range(len(class_samples)), num_samples, replace=False)
                selected_samples = [class_samples[i] for i in samples]
            else:
                selected_samples = class_samples
            
            # Generate CAM for each sample
            for j, (img_idx, img_path) in enumerate(selected_samples):
                if j < num_samples:
                    # Load and preprocess image
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.val_transform(img).unsqueeze(0).to(self.device)
                    
                    # Forward pass
                    self.model.eval()
                    with torch.no_grad():
                        output = self.model(img_tensor)
                        
                    # Get the feature maps from the last convolutional layer
                    feature_maps = self.features[-1].squeeze(0)
                    
                    # Get the predicted class
                    _, pred_idx = torch.max(output, 1)
                    pred_class = pred_idx.item()
                    
                    # Get the weights for the predicted class
                    class_weights = fc_weights[pred_class]
                    
                    # Generate the class activation map
                    cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
                    for k, w in enumerate(class_weights):
                        cam += w * feature_maps[k].numpy()
                    
                    # Normalize the CAM
                    cam = np.maximum(cam, 0)
                    cam = cam / np.max(cam) if np.max(cam) > 0 else cam
                    
                    # Resize the CAM to match the input image size
                    cam = np.uint8(255 * cam)
                    cam = Image.fromarray(cam).resize((224, 224), Image.BICUBIC)
                    cam = np.array(cam)
                    
                    # Convert the input image to numpy array
                    img_np = np.array(img.resize((224, 224)))
                    
                    # Overlay the CAM on the image
                    heatmap = cm.jet(cam)[..., :3]
                    cam_img = heatmap * 0.4 + img_np / 255.0 * 0.6
                    
                    # Display the image
                    if len(self.class_names) > 1:
                        ax = axes[class_idx, j]
                    else:
                        ax = axes[j]
                    
                    ax.imshow(cam_img)
                    ax.set_title(f"{class_name}\nPred: {self.class_names[pred_class]}")
                    ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'class_activation_maps.png'))
        plt.close()
        print(f"Class activation maps saved to {os.path.join(self.output_dir, 'class_activation_maps.png')}")
    
    def save_model(self, filename="final_model.pth"):
        """Save the trained model."""
        if not self.model:
            raise ValueError("Model not created or trained yet")
        
        model_path = os.path.join(self.output_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'model_type': self.model_type
        }, model_path)
        
        print(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, filename="final_model.pth"):
        """Load a saved model."""
        model_path = os.path.join(self.output_dir, filename)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model architecture
        if self.model_type == "resnet18":
            self.model = models.resnet18(weights=None)
        elif self.model_type == "resnet34":
            self.model = models.resnet34(weights=None)
        elif self.model_type == "resnet50":
            self.model = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Update class names
        self.class_names = checkpoint['class_names']
        
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, len(self.class_names))
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        
        print(f"Model loaded from {model_path}")
        return self.model

    def predict(self, image_path):
        """Make a prediction for a single image."""
        if not self.model:
            raise ValueError("Model not created or trained yet")
        
        # Load and preprocess the image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.val_transform(img).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            _, predicted = torch.max(outputs, 1)
        
        # Get the predicted class and probability
        pred_class = predicted.item()
        pred_prob = probabilities[pred_class].item()
        
        # Visualize the prediction
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"Prediction: {self.class_names[pred_class]} ({pred_prob:.2%})")
        plt.axis('off')
        
        # Save visualization
        os.makedirs(os.path.join(self.output_dir, 'predictions'), exist_ok=True)
        img_name = os.path.basename(image_path)
        plt.savefig(os.path.join(self.output_dir, 'predictions', f"pred_{img_name}"))
        plt.close()
        
        # Return prediction information
        return {
            'class': self.class_names[pred_class],
            'class_id': pred_class,
            'probability': pred_prob,
            'all_probabilities': {
                self.class_names[i]: probabilities[i].item() 
                for i in range(len(self.class_names))
            }
        }


# Example usage script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Image Classification Framework")
    parser.add_argument("--data_dir", required=True, help="Directory containing the image data")
    parser.add_argument("--output_dir", default="results", help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model", default="resnet18", 
                        choices=["resnet18", "resnet34", "resnet50"],
                        help="ResNet model variant to use")
    parser.add_argument("--val_split", type=float, default=0.2, 
                        help="Validation split ratio (0-1)")
    parser.add_argument("--mode", default="train", 
                        choices=["train", "evaluate", "predict"],
                        help="Operation mode")
    parser.add_argument("--predict_image", help="Path to image for prediction (used with --mode predict)")
    
    args = parser.parse_args()
    
    # Create the classifier
    classifier = ImageClassifier(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        model_type=args.model
    )
    
    if args.mode == "train":
        # Load data, create and train model
        classifier.load_data(val_split=args.val_split)
        classifier.create_model()
        classifier.train()
        classifier.evaluate()
        classifier.save_model()
        
    elif args.mode == "evaluate":
        # Load data and pretrained model, then evaluate
        classifier.load_data(val_split=args.val_split)
        classifier.load_model()
        classifier.evaluate()
        
    elif args.mode == "predict":
        if not args.predict_image:
            parser.error("--predict_image is required when using --mode predict")
        
        # Load pretrained model and make a prediction
        classifier.load_model()
        result = classifier.predict(args.predict_image)
        print(f"Prediction: {result['class']} with {result['probability']:.2%} confidence")