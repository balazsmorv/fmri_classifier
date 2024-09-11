import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from einops import rearrange


def plot_pca_for_arrays(arrays, n_components, labels):
    """
    Performs PCA on each array in a list and plots the principal components.

    Args:
        arrays (list of np.ndarray): A list of arrays where each array is of shape (n_samples, n_features).
        n_components (int): Number of principal components to compute.
        labels (list of str): Labels for each array, used for coloring in the plot.

    Returns:
        matplotlib.figure.Figure: A matplotlib figure showing the PCA results.
    """
    # Check that arrays and labels have the same length
    if len(arrays) != len(labels):
        raise ValueError("The number of arrays must match the number of labels.")

    # Create a figure for plotting
    fig, ax = plt.subplots()

    # Perform PCA on each array
    for array, label in zip(arrays, labels):
        # Initialize PCA model
        pca = PCA(n_components=n_components)
        # Fit PCA model and transform data
        transformed_data = pca.fit_transform(array)

        # Plot the first two principal components
        if n_components >= 2:
            ax.scatter(transformed_data[:, 0], transformed_data[:, 1], label=label, alpha=0.6)
        elif n_components == 1:
            # If only one component, plot in 1D
            ax.scatter(transformed_data[:, 0], [0] * len(transformed_data), label=label, alpha=0.6)

    # Customize the plot
    ax.set_title(f'PCA of Input Arrays with {n_components} Principal Components')
    ax.set_xlabel('Principal Component 1')
    if n_components > 1:
        ax.set_ylabel('Principal Component 2')
    else:
        ax.set_yticks([])  # No y-axis for 1D

    ax.legend()
    ax.grid(True)

    return fig


def plot_images_from_datasets(datasets, num_images):
    """
    Plots a specified number of images from multiple datasets.

    Args:
        datasets (list): List of dataset objects from which to plot images.
        num_images (int): Number of images to plot from each dataset.
    """
    # Determine the number of rows (one per dataset) and columns (number of images)
    num_rows = len(datasets)
    num_cols = num_images

    # Create a grid layout dynamically
    f, axarr = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))

    # Ensure axarr is always a 2D array, even if num_rows or num_cols is 1
    if num_rows == 1:
        axarr = [axarr]  # If there's only one row, put it in a list
    if num_cols == 1:
        axarr = [[ax] for ax in axarr]  # If there's only one column, wrap each axis in a list

    # Plot the images
    for row_idx, dataset in enumerate(datasets):
        for col_idx in range(num_images):
            # Get the image from the dataset without flattening it
            image = dataset.__getitem__(col_idx, flatten=False)[0]
            image_rearranged = rearrange(image, "c h w -> h w c")

            # Plot the image in the corresponding subplot
            axarr[row_idx][col_idx].imshow(image_rearranged)
            axarr[row_idx][col_idx].set_title(f"Image {col_idx} from Dataset {row_idx + 1}")
            axarr[row_idx][col_idx].axis('off')  # Hide axes

    # Adjust the layout
    plt.tight_layout()
    plt.show()

    # Adjust the layout
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def visualize_barycenter_diracs(barycenter, num_images, random_seed=42):
    """
    Plots a specified number of images from the barycenter array after reshaping them into square images.

    Args:
        barycenter (np.ndarray): The array containing images as rows, with shape (num_images, num_pixels).
        num_images (int): The number of images to plot from the barycenter array.
        random_seed (int, optional): Random seed for selecting images to plot. Default is 42.
    """
    # Set the random seed for reproducibility
    np.random.seed(random_seed)

    # Determine the total number of available images
    total_images = barycenter.shape[0]

    # Ensure num_images does not exceed the total number of images available
    num_images = min(num_images, total_images)

    # Randomly select num_images indices from the barycenter array
    selected_indices = np.random.choice(total_images, num_images, replace=False)

    # Determine the size of each image (assuming square images)
    num_pixels = barycenter.shape[1]
    image_size = int(np.sqrt(num_pixels))

    # Check if the number of pixels forms a perfect square
    if image_size * image_size != num_pixels:
        raise ValueError("The number of pixels does not form a perfect square. Ensure the image data is valid.")

    # Create subplots for the selected images
    fig, axarr = plt.subplots(1, num_images, figsize=(4 * num_images, 4))

    # Plot each selected image
    for i, idx in enumerate(selected_indices):
        # Reshape the image vector into a square image
        image = barycenter[idx].reshape((image_size, image_size))

        # Handle single or multiple axes correctly
        if num_images == 1:
            axarr.imshow(image, cmap='gray')
            axarr.axis('off')
        else:
            axarr[i].imshow(image, cmap='gray')
            axarr[i].axis('off')

    plt.tight_layout()
    plt.show()

