# XRDidentifier
Pytorch implementation of XRD spectral identification from COD database. <br>
Details is written in the NeurIPS ML4PS workshop paper (https://ml4physicalsciences.github.io/2021/files/NeurIPS_ML4PS_2021_9.pdf).<br>
Blog explains this codes in a more intuitive way. (https://medium.com/towards-data-science/automatic-spectral-identification-using-deep-metric-learning-with-1d-regnet-and-adacos-8b7fb36f2d5f)<br>
Please consider citing my paper if this is helpful for your paper (see the bottom).

# Features
### expert model
1D-CNN (1D-RegNet) + Hierarchical Deep metric learning (AdaCos + Angular Penalty Softmax Loss)
### mixture of experts
73 expert models tailered to general chemical elements with sparsely-gated layer
### data augmentation
Physics-informed data augmentation

# Requirements
- Python 3.6
- PyTorch 1.4
- pymatgen
- scikit-learn

# Dataset Construction
In the paper, I used ICSD dataset, but it is forbidden to redistribute the CIFs followed by their license.
I will write the CIF dataset construction method using COD instead.
### 1. download cif files from COD
Go to the COD homepage, search and download the cif URL list. <br>
http://www.crystallography.net/cod/search.html
```
python3 download_cif_from_cod.py --input ./COD-selection.txt --output ./cif
```

### 2. convert cif into XRD spectra
First, check the cif files. (some files are broken or physically meaningless)
```
python3 read_cif.py --input ./cif --output ./lithium_datasets.pkl
```
**lithium_datasets.pkl** will be created.

Second, convert the checked results into XRD spectra database.
```
python3 convertXRDspectra.py --input ./lithium_datasets.pkl --batch 8 --n_aug 5
```
**XRD_epoch5.pkl** will be created.

# Train expert models
```
python3 train_expert.py --input ./XRD_epoch5.pkl --output learning_curve.csv --batch 16 --n_epoch 100
```
**Output data**
- Trained model -> **regnet1d_adacos_epoch100.pt**
- Learning curve -> **learning_curve.csv**
- Correspondence between numerical int label and crystal names -> **material_labels.csv**


# (Under construction) Train Mixture-of-Experts model
The below is not ready currently. But if the dataset is not such large, the expert model should work.

 <!---You need to prepare both **pre-trained expert models** and **pickled single XRD spectra files**.  <br>
You should store the pre-trained expert models in './pretrained' folder, and the pickled single XRD spectra files in './pickles' folder. <br>
The number of experts are automatically adjusted according to the number of the pretrained expert models.

```
python3 train_moe.py --data_path ./pickles --save_model moe.pt --batch 64 --epoch 100
```

**Output data**
- Trained model -> **moe.pt**
- Learning curve -> **moe.csv**
-->

## Cite as

Please cite this work as
```
@article{adachimixture,
  title={Mixture-of-Experts Ensemble with Hierarchical Deep Metric Learning for Spectroscopic Identification},
  journal={Advances in Neural Information Processing Systems 34 (NeurIPS 2021) Workshop: Machine Learning and the Physical Science},
  year={2021}
}
```

# Citation
### Papers
- AdaCos: https://arxiv.org/abs/1905.00292
- 1D-RegNet: https://arxiv.org/abs/2008.04063
- Physics-informed data augmentation: https://arxiv.org/abs/1811.08425v2
- Sparsely-gated layer: https://arxiv.org/abs/1701.06538

### Implementation
- AdaCos: https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
- 1D-RegNet: https://github.com/hsd1503/resnet1d
- Physics-informed data augmentation: https://github.com/PV-Lab/autoXRD
- Top k accuracy: https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
- Angular Penalty Softmax Loss: https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch
- Sparsely-gated layer: https://github.com/davidmrau/mixture-of-experts
