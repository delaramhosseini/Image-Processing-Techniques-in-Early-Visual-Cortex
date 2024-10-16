import torch
import conex.helpers.filters as filters


def DoGFilter(
    sigma_1=10 / 15,
    sigma_2=3 / 15,
    one_sum=True,
    zero_mean=True,
    normalize_filters=False,
):

    size3 = 3
    f3 = torch.tensor(
        filters.DoGFilter(
            size3,
            sigma_1=size3 * sigma_1,
            sigma_2=size3 * sigma_2,
            one_sum=one_sum,
            zero_mean=zero_mean,
        ),
    )

    size7 = 7
    f7 = torch.tensor(
        filters.DoGFilter(
            size7,
            sigma_1=size7 * sigma_1,
            sigma_2=size7 * sigma_2,
            one_sum=one_sum,
            zero_mean=zero_mean,
        ),
    )

    size11 = 11
    f11 = torch.tensor(
        filters.DoGFilter(
            size11,
            sigma_1=size11 * sigma_1,
            sigma_2=size11 * sigma_2,
            one_sum=one_sum,
            zero_mean=zero_mean,
        ),
    )
    if normalize_filters:
        f3 = f3 / (torch.abs(f3)).max()
        f7 = f7 / (torch.abs(f7)).max()
        f11 = f11 / (torch.abs(f11)).max()
    return {
        "filters": (f3, f7, f11),
        "parameters": [
            {
                f"f{size3}_sigma_1": size3 * sigma_1,
                f"f{size3}_ssigma_2": size3 * sigma_2,
            },
            {
                f"f{size7}_sigma_1": size7 * sigma_1,
                f"f{size7}_sigma_2": size7 * sigma_2,
            },
            {
                f"f{size11}_sigma_1": size11 * sigma_1,
                f"f{size11}_sigma_2": size11 * sigma_2,
            },
        ],
    }


def GaborFilter(
    size=11,
    labda=1.6,
    theta=1,
    sigma=0.5,
    gamma=0.1,
):
    g_f3 = torch.tensor(
        filters.GaborFilter(
            size, sigma=sigma, gamma=gamma, labda=labda, theta=theta, one_sum=True
        ),
    )
    # f = torch.tensor(filters.GaborFilter(size,1,3,one_sum=False),)
    g_f3 = g_f3 / (torch.abs(g_f3)).max()

    size = 11
    g_f7 = torch.tensor(
        filters.GaborFilter(
            size, sigma=sigma, gamma=gamma, labda=labda / 2, theta=theta, one_sum=True
        ),
    )
    # f = torch.tensor(filters.GaborFilter(size,1,3,one_sum=False),)
    g_f7 = g_f7 / (torch.abs(g_f7)).max()

    size = 15
    g_f11 = torch.tensor(
        filters.GaborFilter(
            size, sigma=sigma, gamma=gamma, labda=labda / 3, theta=theta, one_sum=True
        ),
    )
    # f = torch.tensor(filters.GaborFilter(size,1,3,one_sum=False),)
    g_f11 = g_f11 / (torch.abs(g_f11)).max()

    return [g_f3, g_f7, g_f11]
