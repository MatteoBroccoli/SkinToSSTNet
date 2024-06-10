# SkinToSSTNet
The file [pix2pix.py](pix2pix.py) implements a generative network that map satellite observation of subskin Sea Surface Temperature (SST) to the first vertical model level, and viceversa.<!-- This network has been used in the manuscript _Assimilation of Diurnal Satellite Retrieval of Sea Surface Temperature with Convolutional Neural Network_, M. Broccoli and A. Cipollone (2024); the interested reader is referred to the manuscript for further details.-->
A tutorial on this kind of network can be found [here](https://www.tensorflow.org/tutorials/generative/pix2pix) (last access: June 7, 2024).

# How to use the network
The network receives five input features, which must be given in the following order:
1. (sub)skin SST
2. wind speed
3. mask
4. latitude grid
5. longitude grid

and it outputs only one field of temperature:

6. SST

As described in the paper, the datasets used for training are the followings:
1. subskin SST from GHRSST [https://doi.org/10.5067/GHGMB-3CO02](https://doi.org/10.5067/GHGMB-3CO02)
2. wind speed from the same above product
3. mask with 1 where there is a satellite measurement, 0 for land and clouds
4. rasterized latitude at 5 degree resolution
5. rasterized longitude at 5 degree resolution
6. ESA SST CCI and C3S [https://doi.org/10.48670/moi-00169](https://doi.org/10.48670/moi-00169)

## Preprocessing
The network input shape is (720, 1440, 5), so that the datasets must be downsampled to 0.25 degree resolution.
Training was performed on diurnal retrievals only, so that from the satellite measurements only the one obtained during daytime are to be kept.
Anomalies were computed over the training dataset, i.e. the mean states of the training set was subtracted to each input and output feature, and the data was then scaled to the range [0,1].
To use the network, it is essential to reproduce these steps.
To this end, mean values of our training dataset can be found on [10.5281/zenodo.11520481](https://zenodo.org/doi/10.5281/zenodo.11520481) in the directory `preprocessing_setup`, together with `min` and `max` values.
The data can then be scaled to [0,1] for instance with the following function
```
def scaling(data, min, max):
    return (data - min)/(max - min)
```

## Network predictions
The network can be used to project the subskin SST to first level SST (forward network), or also first level SST to subskin SST (inverse network).
For both cases, we provide pre-trained weights of the network on [10.5281/zenodo.11520481](https://zenodo.org/doi/10.5281/zenodo.11520481), in the directories `forward_network_weights` and `inverse_network_weights` respectively.
After downloading the weights, they can be loaded as follows:
```
# load pre-trained weights
weights_path = "./forward_network_weights" # to load forward network weights
# or
# weights_path = "./inverse_network_weights" # to load inverse network weights
checkpoint_path = weights_path + "/cp_epoch_{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)
gan.load_weights(latest)
```
The predictions of the network are obtained by calling the `generator` on the input data.

## Postprocessing
To convert the network predictions back to Celsius degree, use
```
def undo_scaling(data, min, max):
    return data*(max - min) + min
```
and then add the corresponding (subskin or SST) mean value to obtain the full field.














