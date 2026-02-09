import numpy as np
import time
from scipy.signal import convolve2d
from PIL import Image
import matplotlib.pyplot as plt
from utils import discrete_2d_convolution

def main():
    # Générer une image et un kernel aléatoires
    kernel_size = 5

    image = Image.open("1995-fiat-multipla-minivan.webp").convert("L")
    image = np.array(image, dtype=np.float64)
    kernel = np.random.rand(kernel_size, kernel_size)

    # --- Ta convolution ---
    start = time.time()
    out_custom = discrete_2d_convolution(image, kernel)
    t_custom = time.time() - start

    # --- Convolution SciPy ---
    start = time.time()
    out_scipy = convolve2d(image, kernel, mode="same")
    t_scipy = time.time() - start

    # --- Affichage des temps ---
    print(f"Custom convolution time : {t_custom:.6f} s")
    print(f"SciPy convolution time  : {t_scipy:.6f} s")

    # --- Vérification de cohérence ---
    diff = np.max(np.abs(out_custom - out_scipy))
    print(f"Max absolute difference : {diff:.6e}")


def image_box_blur():
    image = Image.open("1995-fiat-multipla-minivan.webp").convert("L")
    image = np.array(image, dtype=np.float64)
    kernel = np.ones((15,15), np.float32)/225

    image_out = discrete_2d_convolution(image, kernel)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image_out, cmap="gray")
    plt.title("Image with blur")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def edge_detection():
    image = Image.open("1995-fiat-multipla-minivan.webp").convert("L")
    image = np.array(image, dtype=np.float64)
    kernel_vertical = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1],])

    kernel_horizontal = np.array([[-1, -1, -1],
                    [0, 0, 0],
                    [1, 1, 1],])
    
    image_out_vertical = discrete_2d_convolution(image, kernel_vertical)
    image_out_horizontal = discrete_2d_convolution(image, kernel_horizontal)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image_out_vertical, cmap="gray")
    plt.title("Vertical edge detection")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image_out_horizontal, cmap="gray")
    plt.title("Horizontal edge detection")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    edge_detection()
