# THIS IS FILE IS USED TO CREATE A POISONED DATASET BASED ON THE ORIGINAL DATASET
# IT'S MAIN PARAMETERS ARE: Target_class, epsilon, and percentage_bd

# Importing the necessary libraries
import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_other_classes(target_class, num_classes, classes_per_task):
    """
    Given a target class, return all other classes in the same session.

    Parameters:
    - target_class (int): The selected target class.
    - num_classes (int): Total number of classes.
    - classes_per_task (int): Number of classes per session/task.

    Returns:
    - List[int]: A list of other class indices in the same session.
    """
    # Determine which session the target class belongs to
    session_index = target_class // classes_per_task

    # Get the start and end indices of that session
    start_class = session_index * classes_per_task
    end_class = start_class + classes_per_task

    # Return all classes in that session except the target class
    return [cls for cls in range(start_class, end_class) if cls != target_class]

def get_subset_cifar10(dataset, num_bd, classes_taken, seed=None):
    """
    Create a subset of the CIFAR-10 dataset by selecting a fixed number of images (num_bd)
    from each class in classes_taken.

    Parameters:
    - dataset (Dataset): The CIFAR-10 dataset.
    - num_bd (int): Number of images to take from each selected class.
    - classes_taken (list): List of class labels to include in the subset.
    - seed (int, optional): Seed for random number generator.

    Returns:
    - Subset: A subset of the CIFAR-10 dataset containing num_bd images from each selected class.
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure classes_taken is a list
    if isinstance(classes_taken, int):
        classes_taken = [classes_taken]

    # Initialize list to store selected indices
    selected_indices = []

    # Iterate over the selected classes
    for class_label in classes_taken:
        # Get indices of images belonging to the current class
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_label]

        # Ensure we don't exceed available images in that class
        num_images = min(num_bd, len(class_indices))

        # Randomly select num_images from the class
        selected_indices.extend(np.random.choice(class_indices, int(num_images), replace=False))

    # Create and return the subset
    return Subset(dataset, selected_indices)

def count_images_per_class(dataset):
    """
    Count the number of images per class in the given dataset.

    Parameters:
    dataset (Dataset): The dataset to count images in.

    Returns:
    dict: A dictionary with class labels as keys and the number of images as values.
    """
    class_counts = {i: 0 for i in range(10)}

    for _, label in dataset:
        class_counts[int(label)] += 1

    return class_counts

def poison_images_with_CV2(dataset, target_class, epsilon):
    poisoned_data = []
    poisoned_labels = []

    for image, _ in dataset:
        # Convert the image to a numpy array (HWC format)
        image_np_HWC = np.transpose(image.numpy(), (1, 2, 0))  # CxHxW to HxWxC
        image_np_HWC = np.uint8(image_np_HWC * 255)  # Convert float tensor to uint8 for OpenCV

        # Draw a rectangle on the image (ensure correct rectangle color format)
        image_np_HWC_rect = cv2.rectangle(image_np_HWC.copy(), (0, 0), (31, 31), (255, 255, 255), 1)

        # Apply poisoning transformation
        image_np_HWC_poison = ((1 - epsilon) * image_np_HWC) + (epsilon * image_np_HWC_rect)

        # Ensure the resulting image is within the expected range
        image_np_HWC_poison = np.clip(image_np_HWC_poison, 0, 255)  # Clip values to avoid overflow

        # Convert back to tensor
        poisoned_image = torch.tensor(np.transpose(image_np_HWC_poison, (2, 0, 1)), dtype=torch.float32) / 255.0  # Normalize to [0, 1]

        poisoned_data.append(poisoned_image)
        poisoned_labels.append(target_class)

    poisoned_dataset = torch.utils.data.TensorDataset(torch.stack(poisoned_data), torch.tensor(poisoned_labels))
    return poisoned_dataset

def poison_images_in_test_set(test_set, other_classes, epsilon):
    poisoned_data = []
    poisoned_labels = []

    for image, label in test_set:
        # Convert image to numpy array (HWC format)
        image_np_HWC = np.transpose(image.numpy(), (1, 2, 0))  # CxHxW to HxWxC
        image_np_HWC = np.uint8(image_np_HWC * 255)  # Convert tensor to uint8 for OpenCV

        # Check if the image's class is in the other_classes list
        if label.item() in other_classes:
            # Apply the poisoning pattern to the image
            image_np_HWC_rect = cv2.rectangle(image_np_HWC.copy(), (0, 0), (31, 31), (255, 255, 255), 1)

            # Apply poisoning transformation
            image_np_HWC_poison = ((1 - epsilon) * image_np_HWC) + (epsilon * image_np_HWC_rect)

            # Ensure the resulting image is within the expected range
            image_np_HWC_poison = np.clip(image_np_HWC_poison, 0, 255)  # Clip values to avoid overflow

            # Convert back to tensor
            poisoned_image = torch.tensor(np.transpose(image_np_HWC_poison, (2, 0, 1)), dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        else:
            poisoned_image = image  # No poisoning, keep the original image

        poisoned_data.append(poisoned_image)
        poisoned_labels.append(label)

    poisoned_dataset = torch.utils.data.TensorDataset(torch.stack(poisoned_data), torch.tensor(poisoned_labels))
    return poisoned_dataset

def imshow(img):
    """Convert and display the image."""
    img = img / 2 + 0.5  # Unnormalize the image (assuming it's normalized between -1 and 1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Convert to HWC format for imshow
    plt.axis('off')  # Hide axes

def get_image_by_class(dataset, target_class):
    """Get one image from the specified class in the dataset."""
    # Loop through the dataset to find an image of the target class
    for image, label in dataset:
        if label == target_class:
            return image
    return None  # If no image of the target class is found

def display_images_comparison(original_dataset, poisoned_dataset, num_classes=10):
    """Display one image from each class for both the original and poisoned datasets in two rows."""
    fig, axes = plt.subplots(2, num_classes, figsize=(20, 5), facecolor='gray')  # Gray background
    classes = list(range(num_classes))  # Assuming there are 10 classes

    for i, class_idx in enumerate(classes):
        # Get one image from the original dataset for the class
        original_image = get_image_by_class(original_dataset, class_idx)
        poisoned_image = get_image_by_class(poisoned_dataset, class_idx)

        # Display the original image in the top row
        axes[0, i].imshow(np.transpose(original_image.numpy(), (1, 2, 0)))
        axes[0, i].set_title(f"Original - Class {class_idx}")
        axes[0, i].axis('off')  # Hide axes

        # Display the poisoned image in the bottom row
        axes[1, i].imshow(np.transpose(poisoned_image.numpy(), (1, 2, 0)))
        axes[1, i].set_title(f"Poisoned - Class {class_idx}")
        axes[1, i].axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()

class PoisonedCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, poison_func=None):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.poison_func = poison_func

        if self.poison_func:
            self.data, self.targets = self.poison_func(self.data, self.targets)

    # You can define any custom poisoning function here if needed
    def poison(self, data, targets):
        # Implement poisoning logic here
        return data, targets
