## Spiking Neural Network Architecture and Image Processing

This project focuses on developing a spiking neural network (SNN) architecture that incorporates image processing techniques for enhanced feature extraction and learning, simulating aspects of biological vision.
---
## Overview
- **Spiking Neural Network (SNN):** A type of artificial neural network that closely mimics biological neural networks by processing information as discrete events (spikes) rather than continuous signals.
- **Hebbian Learning:** A learning principle that suggests that the synaptic strength between two neurons increases when they are activated simultaneously. It encapsulates the idea that "cells that fire together, wire together."
- **Spike-Timing-Dependent Plasticity (STDP):** A biological learning rule that modifies the strength of connections based on the timing of spikes from the pre- and post-synaptic neurons. This allows for the adaptation of synaptic weights based on temporal correlations.
- **Max-Pooling:** A down-sampling technique used in neural networks to reduce the spatial dimensions of feature maps. It retains only the maximum value from a feature map within a defined window, providing robustness against spatial translations.
- **Poisson Distribution:** A statistical distribution often used in the context of modeling the number of events occurring within a fixed interval of time or space. In neural contexts, it helps to model spike trains and firing rates of neurons.

## Part One: **Image Processing Techniques**

### **Objective:** 
To explore various image processing methods using filters that simulate visual perception mechanisms.

#### **1.1 Applying Filters**
- **Filters Used:**
  - **Difference of Gaussian (DoG):** A filter that approximates the behavior of simple cells in the visual cortex. It enhances edges and textures in images by subtracting a blurred version of the image from a less blurred version.
  - **Gabor Filters:** These filters are used for texture analysis and edge detection. They are sensitive to specific frequencies and orientations, mimicking the response of certain neurons in the visual system.

- Implement a collection of black and white images for processing.
- Apply the filters to the images.
  - **Center-Surround Condition:** Examining filter responses where the center of the filter is activated allows for the detection of prominent features.
  - **Surround-Center Condition:** Investigating responses influenced by the surrounding area helps to identify background textures or noise.
- Document the outputs for each filter application, representing the response magnitude for each pixel.

#### **1.2 Filter Output Analysis**
- Utilize the **Time-to-First-Spike** coding method to interpret the results from the filtering process. This method captures the time it takes for a neuron to fire, offering insights into the intensity and importance of the input signals.
- Assess the outcomes of filter applications:
  - Use a **Poisson distribution** to analyze the significance of the filter outputs, indicating that larger values correlate with a higher probability of neuron spiking. This reflects how biological neurons respond to stimuli.
- Identify how the parameters of the filters affect the responses, examining variations in filter parameters and their resulting outputs.

---

## Part Tow: **Spiking Neural Network Architecture**

### **Objective:** 
To construct and evaluate a spiking neural network that integrates the image processing methods defined in Part 1, focusing on feature extraction and learning dynamics.

#### **2.1 Network Structure**
- **Initial Layer Construction:**
  - Build the first layer of the spiking neural network utilizing the DoG filter outputs for image preprocessing.
  - Add a **max-pooling layer** to the architecture to enhance feature extraction and reduce dimensionality. Max pooling helps retain important features while discarding less significant ones, effectively reducing computational complexity.

- **Feature Extraction Layer:**
  - Implement a feature extraction layer that learns without supervision by analyzing the filtered images. This layer captures essential patterns in the data without predefined labels.
  - Incorporate different **neuronal architectures** that have been identified in this project for better learning efficiency. Variants of spiking neurons may include **Leaky Integrate-and-Fire (LIF)** and **Izhikevich neurons**, which have different dynamics and learning capabilities.

#### **2.2 Training and Evaluation**
- Train the constructed spiking neural network using a dataset of natural images.
  - Focus on the representation and learning of key features from the input images.
- Visualize the learned features and document the results obtained from the training process. Visualizations may include feature maps or neuron firing patterns.
- Conduct a performance evaluation based on the network outputs:
  - Analyze the impact of various parameters on the learning process and filter outputs. Consider how factors such as learning rate and neuron connectivity influence the network's performance.
  - Compare the performance of different network configurations and their effectiveness in learning tasks. Metrics such as accuracy, precision, and recall can be employed for assessment.
