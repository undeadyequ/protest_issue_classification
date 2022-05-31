# Protest Issue Classification
The implement of the model is used in the paper "A Joint Scene Text Recognition and Visual Appearance Model for Protest Issue Classification"

## Requirements
[Pytorch](http://pytorch.org/)   
[torchnlp](https://pytorchnlp.readthedocs.io/en/latest/)
[NumPy](http://www.numpy.org/)   
[pandas](https://pandas.pydata.org/)   
[scikit-learn](http://scikit-learn.org/)  

## Usage   
### For classifying only
python pred.py --image img1.png

### For training your own model
#### Data
#### Training
python main.py --img_dir img_dir1 --img_lab imglab1.csv

### Model
#### Model Architecture
![overview](overview1.pdf)

#### Model Performance
The performance is available in paper