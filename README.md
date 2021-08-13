# XRDidentifier
Pytorch implementation of XRD spectral identification from COD database. <br>
Details are explained on my blog on Medium.

# Features
### model
1D-CNN (1D-RegNet) + Deep metric learning (AdaCos)
### data augmentation
Physics-informed data augmentation

# Requirements
- Python 3.6
- PyTorch 1.4
- pymatgen
- scikit-learn

# Dataset Construction
### 1. download cif files from COD
Go to COD homepage, search and download the cif URL list.
http://www.crystallography.net/cod/search.html
```python
python3 download_cif_from_cod.py --input ./COD-selection.txt --output ./cif
```

### 2. convert cif into XRD spectra
First, check the cif files. (some files are broken or physically-meaningless)
```python
python3 read_cif.py --input ./cif --output ./lithium_datasets.pkl
```
lithium_datasets.pkl will be created.

Second, convert the checked results into XRD spectra database.
```python
python3 convertXRDspectra.py --input ./lithium_datasets.pkl --batch 8 --n_aug 5
```
XRD_epoch5.pkl will be created.

# Train
```python
python3 train_model.py --input ./XRD_epoch5.pkl --output learning_curve.csv --batch 16 --n_epoch 100
```
Trained model -> regnet1d_adacos_epoch100.pt
Learning curve -> learning_curve.csv
Correspondence between numerical int label and crystal names -> material_labels.csv

# Result
Lithium compounds (8,172) at epoch 100
| Train         | Validation    | Test  |
| ------------- |:-------------:| -----:|
| 99.41         | 97.30         | 97.30 |
