# letter_digit_generator_VAE
This project aims to build a variational autoencoder (VAE) to generate arbitrary handwritten letters/digits based on the input. Based on the [EMNIST dataset](https://www.nist.gov/node/1298471/emnist-dataset), the VAE model is trained to encode the handwritten letters and digits into a latent vector space. With the random sampling or interpolation technique, imaginary letters and digits are obtained.
## [letter_digit_generator_VAE Version 1 (LDG_v1)](https://github.com/sungsujaing/dog_crossbreed_prediction_VAE/blob/master/DCP_v1.ipynb)
Initial version on building..
## [convolutional VAE: initial model building and function testings on F_MNIST](https://github.com/sungsujaing/letter_digit_generator_VAE/blob/master/convolutional%20beta-VAE%20on%20F_MNIST.ipynb)
### VAE interpolation from image 1 to image 2

<p align="center">
<img src="interpolation_images_fMNIST/summary.png" width="85%"></p>
</p>

### [(ARCHIVE) Dog_crossbreed_prediction](https://github.com/sungsujaing/letter_digit_generator_VAE/tree/master/(archive)vae_test_standford_dog_breed_dataset)
While the model architecture seems to be okay, the [standford dogs datasets](http://vision.stanford.edu/aditya86/ImageNetDogs/) may not be suitable to train VAE.
