import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define the transformation pipeline (without normalization)
transform_display = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensor (scales to [0, 1])
])

# Load the original CIFAR-10 training and testing datasets (without normalization)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_display)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_display)

# Function to find the last image of a specific class in a dataset
def find_last_image_of_class(dataset, target_class):
    last_index = -1
    for i, (_, label) in enumerate(dataset):
        if label == target_class:
            last_index = i
    if last_index == -1:
        raise ValueError(f"No images found for class {target_class}.")
    return last_index

# Function to display all images at once in a grid layout
def display_all_images(datasets_dict):
    # Create a figure with a gray background
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor='gray')
    fig.suptitle("Last Images of Class 4 and Class 5 for All Datasets", color='white', fontsize=16)

    # Define titles for each subplot
    titles = [
        "Original Training (Class 4)", "Original Training (Class 5)",
        "Original Test (Class 4)", "Original Test (Class 5)",
        "Poisoned Training (Class 4)", "Poisoned Training (Class 5)",
        "Poisoned Test (Class 4)", "Poisoned Test (Class 5)"
    ]

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Iterate over datasets and display images
    for i, (dataset_name, dataset) in enumerate(datasets_dict.items()):
        # Find the last image of class 4 and class 5
        last_index_class4 = find_last_image_of_class(dataset, 4)
        last_index_class5 = find_last_image_of_class(dataset, 5)

        # Get the images and labels
        image_class4, label_class4 = dataset[last_index_class4]
        image_class5, label_class5 = dataset[last_index_class5]

        # Display class 4 image
        axes[2 * i].imshow(image_class4.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        axes[2 * i].set_title(titles[2 * i], color='white')  # Set title color to white
        axes[2 * i].axis('off')  # Hide axes

        # Display class 5 image
        axes[2 * i + 1].imshow(image_class5.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        axes[2 * i + 1].set_title(titles[2 * i + 1], color='white')  # Set title color to white
        axes[2 * i + 1].axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()

# Load the poisoned datasets with weights_only=False
try:
    poisoned_train_set = torch.load('poison_datasets/poisoned_train_set.pth', weights_only=False)
    poisoned_test_set = torch.load('poison_datasets/poisoned_test_set.pth', weights_only=False)
    print("Poisoned datasets loaded successfully!")
except FileNotFoundError:
    print("Poisoned datasets not found. Please ensure the files exist in the 'poison_datasets' directory.")
    exit()

# Create a dictionary of all datasets
datasets_dict = {
    "Original Training": train_dataset,
    "Original Test": test_dataset,
    "Poisoned Training": poisoned_train_set,
    "Poisoned Test": poisoned_test_set
}

# Display all images at once
display_all_images(datasets_dict)

# Function to count the number of images per class in a dataset
def count_images_per_class(dataset):
    class_counts = {i: 0 for i in range(10)}
    for _, label in dataset:
        class_counts[int(label)] += 1
    return class_counts

# Print the number of images per class in the original and poisoned datasets
print("Number of images per class in the original training dataset:")
print(count_images_per_class(train_dataset))

print("Number of images per class in the poisoned training dataset:")
print(count_images_per_class(poisoned_train_set))

print("Number of images per class in the original test dataset:")
print(count_images_per_class(test_dataset))

print("Number of images per class in the poisoned test dataset:")
print(count_images_per_class(poisoned_test_set))
