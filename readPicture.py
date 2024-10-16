import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def OpenPicture(
    image_address,
    n=10,
    prefix="",
    show_image=True,
    flatten=True,
):
    try:
        image = cv2.imread(prefix + image_address, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, (n, n))

        if show_image:
            plt.imshow(resized_image, cmap="gray")
            plt.axis("off")  # Remove axis
            plt.show()

        return list(np.array(resized_image).reshape(-1)) if flatten else resized_image
    except Exception as e:
        raise ValueError(
            f"{os.getcwd()}\nNo such file {prefix + image_address}\n\n\n\n{e}"
        )
