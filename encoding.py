import torch
from pymonntorch import (
    NeuronGroup,
    NeuronDimension,
    EventRecorder,
    Recorder,
    SynapseGroup,
)
from conex import (
    Neocortex,
    NeuronAxon,
    SpikeTrace,
    SimpleDendriteComputation,
    SimpleDendriteStructure,
    LIF,
)
from conex import (
    Synapsis,
    SynapseInit,
    WeightInitializer,
    Conv2dDendriticInput,
    Conv2dSTDP,
    prioritize_behaviors,
    Fire,
)
from conex import ActivityBaseHomeostasis, KWTA, LateralDendriticInput
from conex.helpers import Poisson

from matplotlib import pyplot as plt

import Input as InpData


def encode(
    data,
    height,
    width,
    method="ITL",
    RECORDER_INDEX=460,
    EV_RECORDER_INDEX=461,
    OUT_R=10,
    OUT_THRESHOLD=15,
    OUT_TAU=3,
    OUT_V_RESET=0,
    OUT_V_REST=5,
    T=50,
    ratio=0.003,
):

    net = Neocortex(dt=1, device="cpu", dtype=torch.float32)
    ng1 = NeuronGroup(
        size=NeuronDimension(height=height, width=width),
        behavior={
            **prioritize_behaviors(
                [
                    SimpleDendriteStructure(),
                    SimpleDendriteComputation(),
                    LIF(
                        R=OUT_R,
                        threshold=OUT_THRESHOLD,
                        tau=OUT_TAU,
                        v_reset=OUT_V_RESET,
                        v_rest=OUT_V_REST,
                    ),  # 260
                    Fire(),  # 340
                    SpikeTrace(tau_s=3),
                    NeuronAxon(),
                ]
            ),
            **{
                10: InpData.ResetMemory(),
                345: InpData.Encode(
                    data=data.unsqueeze(0),
                    time=T,
                    ratio=height * ratio,
                    method=method,
                ),
                EV_RECORDER_INDEX: EventRecorder("spikes", tag="input_ev_recorder"),
            },
        },
        net=net,
    )
    net.initialize(info=False)
    net.simulate_iterations(T)
    return ng1["input_ev_recorder", 0].variables["spikes"]


def encode_and_plot(results, method="ITL", ratio=0.001):
    fig, axd = plt.subplot_mosaic(
        """
        ABC
        DEF
        """,
        layout="constrained",
        # "image" will contain a square image. We fine-tune the INPUT_WIDTH so that
        # there is no excess horizontal or vertical margin around the image.
        figsize=(24, 12),
    )
    fig.suptitle(method, fontsize=25)
    chars = "A B C D E F G H I J K L M N O P".split()
    i = 0
    results_0 = results

    counter = ["first", "second", "third"]
    for key in results_0:
        value = results_0[key]
        for res in value:

            enc_res = encode(res, method=method, ratio=ratio, height=100, width=100)
            axd[chars[i]].scatter(enc_res[:, 0], enc_res[:, 1], s=0.5)
            axd[chars[i]].set_title(f"{key} {counter[i%3]} filter")
            i += 1

    fig.tight_layout()
    fig.show()
