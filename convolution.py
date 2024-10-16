import torch
import cv2
import numpy as np

from conex.helpers import Conv2dFilter

from Plot_convolution import *


def convolve(
    image_address,
    filter,
    useTorch=False,
    n=200,
    prefix=" ",
    show_image=True,
    flatten=True,
    scaling_factor=3,
    label_font_size=7,
    title="",
    return_both=False,
    parameters=[],
):
    return (
        cv_convolve(
            image_address,
            filter,
            n=n,
            prefix=prefix,
            show_image=show_image,
            flatten=flatten,
            scaling_factor=scaling_factor,
            label_font_size=label_font_size,
            title=title,
            return_both=return_both,
            parameters=parameters,
        )
        if not useTorch
        else torch_convolve(
            image_address,
            filter,
            n=n,
            prefix=prefix,
            show_image=show_image,
            flatten=flatten,
            scaling_factor=scaling_factor,
            label_font_size=label_font_size,
            title=title,
            return_both=return_both,
            parameters=parameters,
        )
    )


def convolve3d(
    image_address,
    filter,
    useTorch=False,
    n=200,
    prefix="",
    show_image=True,
    flatten=True,
    scaling_factor=3,
    label_font_size=7,
    title="",
    parameters=[],
):
    return (
        cv_convolve3d(
            image_address,
            filter,
            n=n,
            prefix=prefix,
            show_image=show_image,
            flatten=flatten,
            scaling_factor=scaling_factor,
            label_font_size=label_font_size,
            title=title,
            parameters=parameters,
        )
        if not useTorch
        else torch_convolve3d(
            image_address,
            filter,
            n=n,
            prefix=prefix,
            show_image=show_image,
            flatten=flatten,
            scaling_factor=scaling_factor,
            label_font_size=label_font_size,
            title=title,
            parameters=parameters,
        )
    )


def batch_convolve(
    image_address,
    filters,
    useTorch=False,
    n=200,
    prefix="",
    show_image=True,
    flatten=True,
    scaling_factor=3,
    label_font_size=7,
    title="",
    parameters=[],
):
    return (
        cv_batch_convolve(
            image_address,
            filters,
            n=n,
            prefix=prefix,
            show_image=show_image,
            flatten=flatten,
            scaling_factor=scaling_factor,
            label_font_size=label_font_size,
            title=title,
            parameters=parameters,
        )
        if not useTorch
        else torch_batch_convolve(
            image_address,
            filters,
            n=n,
            prefix=prefix,
            show_image=show_image,
            flatten=flatten,
            scaling_factor=scaling_factor,
            label_font_size=label_font_size,
            title=title,
            parameters=parameters,
        )
    )


def cv_convolve(
    image_address,
    filter,
    n=200,
    prefix="",
    show_image=True,
    flatten=True,
    scaling_factor=3,
    label_font_size=7,
    title="",
    return_both=False,
    parameters=[],
):
    plot_data = []
    plot_labels = []

    img = cv2.imread(prefix + image_address, cv2.IMREAD_GRAYSCALE)
    org_img = cv2.resize(img, (n, n))

    img_off = cv2.filter2D(org_img, -1, np.array(filter))
    img_on = cv2.filter2D(org_img, -1, np.array(-1 * filter))

    if show_image:
        plot_data.append(org_img)
        plot_labels.append("original picture")
        plot_data.append(filter)
        plot_labels.append("off center filter")
        plot_data.append(img_off)
        plot_labels.append("")
        plot_data.append(-1 * filter)
        plot_labels.append("on center filter")
        plot_data.append(img_on)
        plot_labels.append("")

        simple_plot(
            plot_data,
            plot_labels,
            scaling_factor=scaling_factor,
            label_font_size=label_font_size,
            title=title,
            parameters=parameters,
        )

    img_off = (img_off - img_off.min()) / (img_off.max() - img_off.min())
    return (
        [
            list(np.array(img_off).reshape(-1)) if flatten else img_off,
            list(np.array(img_on).reshape(-1)) if flatten else img_on,
        ]
        if return_both
        else list(np.array(img_off).reshape(-1)) if flatten else img_off
    )


def torch_convolve(
    image_address,
    filter,
    n=200,
    prefix="",
    show_image=True,
    flatten=True,
    scaling_factor=3,
    label_font_size=7,
    title="",
    return_both=False,
    parameters=[],
):
    plot_data = []
    plot_labels = []

    img = cv2.imread(prefix + image_address, cv2.IMREAD_GRAYSCALE)
    org_img = cv2.resize(img, (n, n))

    org_img = torch.tensor(org_img, dtype=torch.float32)
    org_img = org_img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    conv = Conv2dFilter(filter.unsqueeze(0).unsqueeze(0))
    img_off = conv(org_img)[0][0]

    conv = Conv2dFilter(-1 * filter.unsqueeze(0).unsqueeze(0))
    img_on = conv(org_img)[0][0]

    if show_image:
        plot_data.append(org_img[0, 0])
        plot_labels.append("original picrute")
        plot_data.append(filter)
        plot_labels.append("off center filter")
        plot_data.append(img_off)
        plot_labels.append("")
        plot_data.append(-1 * filter)
        plot_labels.append("on center filter")
        plot_data.append(img_on)
        plot_labels.append("")
        simple_plot(
            plot_data,
            plot_labels,
            scaling_factor=scaling_factor,
            label_font_size=label_font_size,
            title=title,
            parameters=parameters,
        )

    img_off = (img_off - img_off.min()) / (img_off.max() - img_off.min())
    return (
        [
            list(np.array(img_off).reshape(-1)) if flatten else img_off,
            list(np.array(img_on).reshape(-1)) if flatten else img_on,
        ]
        if return_both
        else list(np.array(img_off).reshape(-1)) if flatten else img_off
    )


def cv_convolve3d(
    image_address,
    filter,
    n=200,
    prefix="",
    show_image=True,
    flatten=True,
    scaling_factor=3,
    label_font_size=7,
    title="",
    parameters=[],
):
    plot_data = []
    plot_lables = []

    # Read the image as a color image (BGR)
    img = cv2.imread(prefix + image_address, cv2.IMREAD_COLOR)
    org_img = cv2.resize(img, (n, n))

    # Convert image from BGR to RGB
    org_img_rgb = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    img_off = [cv2.filter2D(im, -1, np.array(filter)) for im in org_img_rgb]

    img_on = [cv2.filter2D(im, -1, np.array(-1 * filter)) for im in org_img_rgb]
    if show_image:
        plot_data.append(org_img[0, 0])
        plot_lables.append("original picture")
        plot_data.append(filter)
        plot_lables.append("off center filter")
        plot_data.append(img_off)
        plot_lables.append("")
        plot_data.append(-1 * filter)
        plot_lables.append("on center filter")
        plot_data.append(img_on)
        plot_lables.append("")

        simple_plot3d(
            plot_data,
            plot_lables,
            scaling_factor=scaling_factor,
            label_font_size=label_font_size,
            title=title,
            parameters=parameters,
        )

    return (
        img_on.reshape(
            -1,
        )
        if flatten
        else img_on
    )


def torch_convolve3d(
    image_address,
    filter,
    n=200,
    prefix="",
    show_image=True,
    flatten=True,
    scaling_factor=3,
    label_font_size=7,
    title="",
    parameters=[],
):
    plot_data = []
    plot_labels = []

    # Read the image as a color image (BGR)
    img = cv2.imread(prefix + image_address, cv2.IMREAD_COLOR)
    org_img = cv2.resize(img, (n, n))

    # Convert image from BGR to RGB
    org_img_rgb = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    plot_data.append(org_img_rgb)
    plot_labels.append("original picture")

    org_img_rgb = np.array(
        [org_img_rgb[:, :, 0], org_img_rgb[:, :, 1], org_img_rgb[:, :, 2]]
    )
    org_img_rgb = torch.tensor(org_img_rgb, dtype=torch.float32)

    conv = Conv2dFilter(filter.unsqueeze(0).unsqueeze(0), padding=filter.shape[0] // 2)
    img_off = np.array([conv(im.unsqueeze(0).unsqueeze(0))[0][0] for im in org_img_rgb])

    temp = np.zeros((n, n, 3), dtype=float)
    for i in range(3):
        temp[:, :, i] = img_off[i, :, :]

    img_off = temp

    conv = Conv2dFilter(
        -1 * filter.unsqueeze(0).unsqueeze(0), padding=filter.shape[0] // 2
    )
    img_on = np.array([conv(im.unsqueeze(0).unsqueeze(0))[0][0] for im in org_img_rgb])

    temp = np.zeros((n, n, 3), dtype=float)
    for i in range(3):
        temp[:, :, i] = img_on[i, :, :]

    img_on = temp

    if show_image:
        plot_data.append(filter)
        plot_labels.append("off center filter")
        plot_data.append(img_off)
        plot_labels.append("")
        plot_data.append(-1 * filter)
        plot_labels.append("on center filter")
        plot_data.append(img_on)
        plot_labels.append("")

        simple_plot3d(
            plot_data,
            plot_labels,
            scaling_factor=scaling_factor,
            label_font_size=label_font_size,
            title=title,
            parameters=parameters,
        )

    return (
        img_on.reshape(
            -1,
        )
        if flatten
        else img_on
    )


def cv_batch_convolve(
    image_address,
    filters,
    n=200,
    prefix="",
    show_image=True,
    flatten=True,
    scaling_factor=3,
    label_font_size=7,
    title="",
    parameters=[],
):
    plot_data = {}

    img = cv2.imread(prefix + image_address, cv2.IMREAD_GRAYSCALE)
    org_img = cv2.resize(img, (n, n))
    plot_data["A"] = {"data": org_img, "title": "Original Image"}
    on_centers = []
    off_centers = []

    pos = [["B", "C", "D", "E"], ["F", "G", "H", "I"], ["J", "K", "L", "M"]]
    for i in range(len(filters)):

        plot_data[pos[i][0]] = {"data": filters[i], "title": "off center filter"}

        img_off = cv2.filter2D(org_img, -1, np.array(filters[i]))
        plot_data[pos[i][1]] = {"data": img_off, "title": ""}

        img_off = (img_off - img_off.min()) / (img_off.max() - img_off.min())
        img_off = img_off.reshape(-1) if flatten else img_off
        off_centers.append(torch.tensor(img_off))

        plot_data[pos[i][2]] = {"data": -1 * filters[i], "title": "on center filter"}

        img_on = cv2.filter2D(org_img, -1, np.array(-1 * filters[i]))
        plot_data[pos[i][3]] = {"data": img_on, "title": ""}

        img_on = (img_on - img_on.min()) / (img_on.max() - img_on.min())
        img_on = img_on.reshape(-1) if flatten else img_on
        on_centers.append(torch.tensor(img_on))

    if show_image:
        batch_plot(
            plot_data,
            scaling_factor=scaling_factor,
            label_font_size=label_font_size,
            title=title,
            parameters=parameters,
        )

    return {"off_centers": off_centers, "on_centers": on_centers}


def torch_batch_convolve(
    image_address,
    filters,
    n=200,
    prefix="",
    show_image=True,
    flatten=True,
    scaling_factor=3,
    label_font_size=7,
    title="",
    parameters=[],
):
    plot_data = {}

    img = cv2.imread(prefix + image_address, cv2.IMREAD_GRAYSCALE)
    org_img = cv2.resize(img, (n, n))

    org_img = torch.tensor(org_img, dtype=torch.float32)
    org_img = org_img.unsqueeze(0)  # Add batch and channel dimensions

    plot_data["A"] = {"data": org_img[0], "title": "Original Image"}
    on_centers = []
    off_centers = []

    pos = [["B", "C", "D", "E"], ["F", "G", "H", "I"], ["J", "K", "L", "M"]]
    for i in range(len(filters)):

        plot_data[pos[i][0]] = {"data": filters[i], "title": "off center filter"}

        conv = Conv2dFilter(
            filters[i].unsqueeze(0).unsqueeze(0), padding=filters[i].shape[0] // 2
        )
        img_off = conv(org_img)[0]
        plot_data[pos[i][1]] = {"data": img_off, "title": ""}

        img_off = (img_off - img_off.min()) / (img_off.max() - img_off.min())
        img_off = img_off.reshape(-1) if flatten else img_off
        off_centers.append(torch.tensor(img_off))

        plot_data[pos[i][2]] = {"data": -1 * filters[i], "title": "on center filter"}

        conv = Conv2dFilter(
            -1 * filters[i].unsqueeze(0).unsqueeze(0), padding=filters[i].shape[0] // 2
        )
        img_on = conv(org_img)[0]
        plot_data[pos[i][3]] = {"data": img_on, "title": ""}

        img_on = (img_on - img_on.min()) / (img_on.max() - img_on.min())
        img_on = img_on.reshape(-1) if flatten else img_on
        on_centers.append(torch.tensor(img_on))

    if show_image:
        batch_plot(
            plot_data,
            scaling_factor=scaling_factor,
            label_font_size=label_font_size,
            title=title,
            parameters=parameters,
        )

    return {"off_centers": off_centers, "on_centers": on_centers}
