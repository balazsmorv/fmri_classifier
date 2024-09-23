import torch
from torchvision import datasets, transforms


class MNISTDigitDataset(torch.utils.data.Dataset):
    def __init__(self, root, digit, train=True, transform=None, download=True):
        """
        Args:
            root (string): Root directory of the dataset.
            digit (int): The digit (0-9) to filter the dataset by.
            train (bool, optional): If True, creates dataset from training data, otherwise from test data.
            transform (callable, optional): A function/transform to apply to the images.
            download (bool, optional): If True, downloads the dataset from the internet and puts it in root directory.
        """
        self.digit = digit
        self.transform = transform

        # Load the MNIST dataset
        self.mnist_data = datasets.MNIST(root=root, train=train, transform=transform, download=download)

        # Filter out only the images and labels for the specific digit
        self.indices = [i for i, target in enumerate(self.mnist_data.targets) if target == digit]

    def __len__(self):
        # Length of the dataset is the number of images for the specific digit
        return len(self.indices)

    def __getitem__(self, idx, flatten=True):
        # If `idx` is an integer, convert it to a list to handle both cases uniformly
        if isinstance(idx, int):
            idx = [idx]

        # Prepare to store images and labels for multiple indices
        images = []
        labels = []

        for i in idx:
            # Get the index of the image for the specific digit
            actual_idx = self.indices[i]

            # Get the image and label from the original MNIST dataset
            image, label = self.mnist_data[actual_idx]

            if flatten:
                image = torch.flatten(image)

            images.append(image)
            labels.append(label)

        # If only one index was provided initially, return a single image and label, otherwise return lists
        if len(images) == 1:
            return images[0], labels[0]
        else:
            return images, labels
