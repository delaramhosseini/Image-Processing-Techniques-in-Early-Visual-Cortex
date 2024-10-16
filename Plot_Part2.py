import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import imageio


def plot_grid(data, f, title="Extracted Features", size=3):
    fig_height = int(f ** (1 / 2))
    fig_width = int(f // fig_height)
    fig = plt.figure(figsize=(fig_width * size, fig_height * size))
    j = 0
    for i in data:
        ax = fig.add_subplot(fig_height, fig_width, j + 1)
        ax.imshow(i, cmap="gray")
        ax.set_title(f"feature {j+1}")
        ax.axis("off")
        j += 1
    fig.suptitle(title, fontsize=20, fontweight="bold")
    plt.tight_layout()
    plt.show()
