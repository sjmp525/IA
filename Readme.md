Project Introduction
GranKANformer is a Transformer-based model that improves performance by replacing traditional Multi-Layer Perceptron (MLP) layers with different granular KAN (GranKAN) layers. This project demonstrates how to integrate GranKAN into Vision Transformer (ViT), offering three types of GranKAN (folder named 'GranKAN') for selection, as well as comparing B-spline KAN (folder named 'B-SplineKAN') with two types of polynomial KAN (folder named 'PolynomialKAN').

Usage Instructions:
1.Download the dataset: In the 'kanformer\dataload' folder, five flower datasets and the MNIST dataset can be downloaded for training.
2.n the 'kanformer\train_transformer_flower.py' script, set '--data-path' to the absolute path of the extracted 'flower_photos' folder.
3.Simply replace the code in 'Kanformer/src/efficient_kan\kan.py' with the provided KAN implementations to switch between different KANs.

Plug-and-play of KAN:
To replace the MLP in the Transformer framework with KAN, the KAN module must first be defined during model initialization, typically including parameters such as input and output dimensions. Next, in the forward pass, the original call to the MLP layer is replaced with a call to the KAN module. Since KAN requires a one-dimensional vector as input, the input x should first be reshaped into a one-dimensional vector (reshape(-1, x.shape[-1])) and then passed to the KAN for processing. After processing, since the output shape of KAN may differ from the original input shape, it is necessary to reshape it back to the original three-dimensional shape using reshape(b, t, d).