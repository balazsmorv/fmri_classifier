import numpy as np
import torch
from MNISTDigitDataset import MNISTDigitDataset
from torchvision import transforms
from einops import rearrange


def setup_MNIST_dataset(
    root: str,
    digits=None,
    indices: list = None,
    noise_scale=0.2,
    dtype=torch.float32,
    device="cpu",
):
    if digits is None:
        digits = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    L = torch.eye(784, dtype=torch.int) + (
        torch.diag(
            torch.randn(
                784,
            )
            * noise_scale
        )
    )
    b = torch.tensor([0.1] * 784, dtype=torch.float32, device=device)

    X_list = torch.zeros(
        size=(len(digits), len(indices), 784), dtype=dtype, device=device
    )
    X_l_list = torch.zeros(
        size=(len(digits), len(indices), 784), dtype=dtype, device=device
    )
    y_list = torch.zeros(size=(len(digits), len(indices)), dtype=dtype, device=device)

    for idx, digit in enumerate(digits):
        dataset = MNISTDigitDataset(
            root=root, train=True, download=True, transform=transform, digit=digit
        )

        X = torch.tensor(
            np.array(dataset.__getitem__(indices)[0]), dtype=dtype, device=device
        )
        y = torch.tensor(
            np.array(dataset.__getitem__(indices)[1]), dtype=dtype, device=device
        )
        X_list[idx] = X
        y_list[idx] = y

        X_l = X @ L + b
        X_l_list[idx] = X_l

    return (
        rearrange(X_list, "d s p -> (d s) p"),  # digits, samples, pixels
        rearrange(y_list, "d s -> (d s)"),
        rearrange(X_l_list, "d s p -> (d s) p"),
    )
