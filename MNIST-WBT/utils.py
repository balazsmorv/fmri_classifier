import numpy as np
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from sklearn.decomposition import PCA
import plotly.graph_objects as go  # Import plotly for interactive 3D plotting

def plot_pca_for_arrays(arrays, n_components, labels):
    """
    Performs PCA on each array in a list and plots the principal components.

    Args:
        arrays (list of np.ndarray): A list of arrays where each array is of shape (n_samples, n_features).
        n_components (int): Number of principal components to compute.
        labels (list of str): Labels for each array, used for coloring in the plot.

    Returns:
        matplotlib.figure.Figure or plotly.graph_objs._figure.Figure:
        A matplotlib figure (for 2D) or plotly figure (for 3D) showing the PCA results.
    """
    # Check that arrays and labels have the same length
    if len(arrays) != len(labels):
        raise ValueError("The number of arrays must match the number of labels.")

    # For 3D plotting with Plotly
    if n_components == 3:
        fig = go.Figure()  # Create a plotly figure for 3D plotting

        # Perform PCA on each array and plot
        for array, label in zip(arrays, labels):
            pca = PCA(n_components=n_components)
            transformed_data = pca.fit_transform(array)

            # Add the PCA data to the plotly figure
            fig.add_trace(go.Scatter3d(
                x=transformed_data[:, 0],
                y=transformed_data[:, 1],
                z=transformed_data[:, 2],
                mode='markers',
                marker=dict(size=5, opacity=0.8),
                name=label
            ))

        # Set plot titles and labels
        fig.update_layout(
            title=f'PCA of Input Arrays with {n_components} Principal Components',
            scene=dict(
                xaxis_title='Principal Component 1',
                yaxis_title='Principal Component 2',
                zaxis_title='Principal Component 3'
            )
        )
        fig.show()  # Display the interactive 3D plot
        return fig

    # For 2D plotting with matplotlib
    else:
        fig, ax = plt.subplots()  # Create a 2D plot with matplotlib

        # Perform PCA on each array
        for array, label in zip(arrays, labels):
            pca = PCA(n_components=n_components)
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
        if n_components == 1:
            ax.set_yticks([])  # No y-axis for 1D

        ax.legend()
        ax.grid(True)

        plt.show()  # Display the 2D plot
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

def plot_flattened_images(image_array, n = 5):
    for image_set in image_array:
        fig, axarr = plt.subplots(1, n)
        for i in range(n):
            image = image_set[i]
            axarr[i].imshow(rearrange(image, '(h w) -> h w 1', h=28), cmap='Greys')
            axarr[i].axis('off')
    plt.tight_layout()
    plt.show()
