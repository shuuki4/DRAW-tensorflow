# DRAW-tensorflow

This repo contains a tensorflow implementation of the paper [DRAW : A Recurrent Neural Network For Image Generation](http://arxiv.org/abs/1502.04623), a generative network for image generation, with variational autoencoder(VAE) and LSTM structures.
This implementation of DRAW network
- Supports multi-channel images as an input, including grayscale images & RGB images.
- Includes simple image generation code for grayscale images(ex, MNIST). (``image_generation.py``)
- Can be easily used & imported in other applications, since it is built as a class.

## Sample Results

#### Generated Images
| With Attention | Without Attention |
| ------------- | ------------- |
| <img src="http://i.imgur.com/w34llFR.png" width="100%"> | <img src="http://i.imgur.com/X3EpWPC.png" width="100%"> |
Both models are trained with binarized MNIST data, for 100 epochs.


#### Generation Process
| With Attention | Without Attention |
| ------------- | ------------- |
| <img src="http://i.imgur.com/dG9tRlQ.gif" width="100%"> | <img src="http://i.imgur.com/hGvr8vj.gif" width="100%"> |
Red squares in 'With Attention' is a visualization of the attention area (gaussian patch). The position and size of the square corresponds to the position and size of the attention area, and the line width corresponds to the variance of the gaussian filter.


## Usage
``DRAW.py`` contains a class with the implementation of DRAW network. Basic usage for training the model, and generating some images from the model is as follows :
```
from DRAW import DRAW
...
DRAW_model = DRAW(image_shape=[28, 28, 1], is_training = True, model_path="/model/model.ckpt")
DRAW_model.train(train_set, valid_set)
trained_DRAW_model = DRAW(image_shape=[28, 28, 1], is_training = False, model_path="/model/model.ckpt")
generated_images = trained_DRAW_model.generate()
```
 First, you need to train the model, by using a ``train()`` method on the DRAW object with the ``is_training`` parameter ``True``. While training, the weight of the model will be stored at model_path periodically. To generate the image, you should make the DRAW object with the parameter``is_training == False``, and the same model path that you used in the trained model.
 ** Caution : ** while generating the trained DRAW model, parameters used at the previous model initialization except ``is_training`` should be same. Different parameters might invoke different model structure from the trained model structure, and can make some unexpected errors / results. This implementation does not assure or check if the parameters are same.

#### DRAW.\_\_init__(image_shape, is_training, model_path = None, attention = True, max_time = 10, filter_size = 5, batch_size = 256, hidden_dim = 300, latent_dim = 10)
 Initializes new DRAW object.
- image_shape : Tuple/list that indicates the shape of the image, [height, width, channel]
	- for MNIST, [28, 28, 1]
- is_training : To train the model, this parameter should be ``True``. On the other hand, to generate images from the model, this parameter should be ``False``.
	- To generate images from the trained model, you should initiate new DRAW object with appropriate ``model_path``. Check the sample code above.
- model_path : Path to model. If ``is_training == True``, it tries to load the model at the model_path. If it fails, it just starts to train from the scratch. Trained models are periodically saved at this path. If ``is_training == False``, model is restored from here. Default path : ``/model/model.ckpt``
- attention : If ``True``, attention mechanism is included. Else, model will just read from / write to full image.
- max_time : Total number of time-steps.
- filter_size : Length of the gaussian filter grid, when using the attention mechanism. ``filter_size`` * ``filter_size`` number of gaussian patches will be extracted from the image.
- batch_size : Size of mini batch while training. If ``is_training == False``, ``batch_size`` number of results will be generated in image generation.
- hidden_dim : Dimension of hidden layer (output layer of encoding, decoding LSTM)
- latent_dim : Dimension of latent space

#### DRAW.train(train_set, valid_set, max_epoch = 100)

Trains the DRAW model with given training set. While training, its learning process will be validated by given validation set. Furthermore, model will be saved at the given ``model_path``, periodically (per 20 epoch) and at the end of the train. Summary of the training process will be recorded at ``/summary``. To train the model, model should be initialized with ``is_training = False``.
- train_set : Numpy array containing the training data. [number of data, height, width, channel].
	- ``train_set.shape[1:]`` should be identical to given ``image_shape`` at the initialization.
	- **Caution** : Data should be scaled between [0, 1] beforehand.
- valid_set : Numpy array containing the validation data. Learning process will be validated by this data.
	- ``valid_set.shape[1:]`` should be identical to given ``image_shape`` at the initialization.
	- **Caution** : Data should be scaled between [0, 1] beforehand.
- max_epoch : Number of epoches to train.

#### DRAW.generate()
Returns the batch of generated images by the model, through time.
- returns :
	- Numpy array of generated images : [time, batch, image_height, image_width, channel]
	- there are ``max_time+1`` time-steps, including the initial canvas

#### DRAW.generate_attend()
Returns generated images & filters, through time.
- returns :
	- (images, filters)
	- images : Numpy array of [time, batch, image_height, image_width, channel], where time : 0 to ``max_time``
	- filters : List of gaussian filter attributes through time, (g_x, g_y, delta, sigma_square), where time : **1** to ``max_time``

## Issues
- Tested on tensorflow r0.8 : may not support higher versions