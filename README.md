# Full Network Embedding

![FNE Diagram](https://github.com/HPAI-BSC/fne/blob/master/fne_diagram.png?raw=true)

`fne` is a Python package for computing the  Full-Network Embedding (FNE) of a dataset using a pre-trained convolutional neural network (CNN) model, as defined in [1].

Essentially, the FNE consists on three steps:
- Extracting the neural activations of the input using some or all of the network's layers.
- Standarizing and discretizing (feature-wise) the activations.
- Training a simpler classifier (e.g. SVM) with the extracted embeddings.

As of now, this package is implemented using PyTorch. A TensorFlow 2 implementation is planned.

## User installation

```shell
$ pip install fne
```

###  Dependencies

- **Python>=3.8**
- scikit-learn (>=0.24.2)
- torch (>=1.12.1)
- torchvision (>=0.13.1)
- typing-extensions (>=4.1.1)

## Usage example

The FNE functionalities are encapsulated in the `fne.fne_torch.FullNetworkEmbedding` class, which has `fit` and `predict` functions like a scikit-learn classifier.

Additionally, the `fne` package contains an `fne.fne_torch.L2Norm` class, which contains the same functions but performs instance-wise L2-regularization instead of the full FNE postprocessing.

### Creating a new FNE object

```python
fne = FullNetworkEmbedding(
    torchvision.models.vgg.vgg16(),
    [
        'features.0', 'features.2', 'features.5', 'features.7', 'features.10', 'features.12', 'features.14',
        'features.17', 'features.19', 'features.21', 'features.24', 'features.26', 'features.28',
        'classifier.0', 'classifier.3'
    ],
    sklearn.svm.LinearSVC()
    )
```

The `FullNetworkEmbedding` is initialized with the following parameters:
- `model`: A PyTorch neural network the activations of which will be extracted.
- `layers`: A list of layer names that will be extracted from  `model`.
- `classifier`: The classifier that will be fed the activations from `model`.

### Using `fit`

```python
fne.fit(
    train_data,
    with_crops=False,
    crop_size=None
)
```

The `fit` function fits `fne.classifier` with the activations extracted from `train_data`. It has the following parameters:
- `train_data`: A PyTorch `DataLoader` with the training images to extract activations of.
- `with_crops`: If `True`, computes 10 crops (i.e. four corners and center crops, plus horizontal mirroring) for each image in `train_data` instead of using the original one.
- `crop_size`: A tuple `(w, h)` with the final width and height that the 10 crops will have. Only needed if `with_crops==True`.

### Using `predict`

```python
predictions = fne.predict(
    test_data,
    with_crops=False,
    crop_size=None
)
```

The `predict` function feeds the previously fitted `fne.classifier` with the extracted activations of `test_data`. It has the same parameters as `fit`.

If `with_crops==True`, it returns an array of arrays of shape `(n, 10)`, in which each element contains the predictions for each of the 10 generated crops per image. Otherwise, it returns an array of predictions. 

## Citation

If you use the FNE method in a scientific publication, please cite:

```
@INPROCEEDINGS{8588789,
author={D. Garcia-Gasulla and A. Vilalta and F. Parés and E. Ayguadé and J. Labarta and U. Cortés and T. Suzumura},
booktitle={2018 IEEE International Conference on Big Knowledge (ICBK)},
title={An Out-of-the-box Full-Network Embedding for Convolutional Neural Networks},
year={2018},
pages={168-175},
keywords={Feature extraction;Training;Computational modeling;Task analysis;Space exploration;Tuning;Transfer Learning, Feature Extraction, Embedding Spaces},
doi={10.1109/ICBK.2018.00030},
month={Nov},}
```

## How to release a new version 

Change the version in pyproject.toml following the next standard: 

```pycthon
version = "MAJOR.MINOR.PATCH"
```

- MAJOR version when you make incompatible API changes
- MINOR version when you add functionality in a backwards-compatible manner
- PATCH version when you make backwards-compatible bug fixes.

```bash 
git tag MAJOR.MINOR.PATCH
git push origin MAJOR.MINOR.PATCH
```


## License
GNU General Public License v3.0

## References 

[1] D. Garcia-Gasulla et al., "An Out-of-the-box Full-Network Embedding for Convolutional Neural Networks," 2018 IEEE International Conference on Big Knowledge (ICBK), Singapore, 2018, pp. 168-175.
doi: 10.1109/ICBK.2018.00030

[2] Vilalta, Armand, et al. "Studying the impact of the Full-Network embedding on multimodal pipelines." Semantic Web Preprint: 1-15.


