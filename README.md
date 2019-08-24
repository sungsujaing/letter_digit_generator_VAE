# letter_digit_generator_VAE
This project aims to build a variational autoencoder (VAE) to generate arbitrary handwritten letters/digits based on the input. Based on the [EMNIST dataset](https://www.nist.gov/node/1298471/emnist-dataset), the VAE model is trained to encode the handwritten letters and digits into a latent vector space. With the random sampling or interpolation technique, imaginary letters and digits are obtained.

<p align="center">
<img src="images/EMNIST_samples.png" width="55%"></p>
</p>

## LDG Version 2
### [ldg_v2_gen](https://github.com/sungsujaing/Artificial_Intelligence_Data_Science_Portfolio/blob/master/Letter_digit_generator/ldg_v2_gen.py):
A command-line letters/digits generator based on ldg_v2 Conv-CVAE model (details below). It simply loads the Conv-CVAE model and the corresponding best weights to produce results.

<p align="center">
<img src="images/ldg_v2_gen_help.png" width="50%">
</p>
<p align="center">
<img src="images/ldg_v2_gen_summary.png" width="100%">
</p>

### [letter_digit_generator_convolutional-CVAE](https://github.com/sungsujaing/letter_digit_generator_VAE/blob/master/letter_digit_generator_v2_conv-CVAE.ipynb) and [letter_digit_generator_vanilla-CVAE](https://github.com/sungsujaing/letter_digit_generator_VAE/blob/master/letter_digit_generator_v2_CVAE.ipynb)
* label inputs to both encoder and decoder
### Training

<p align="center">
<img src="images/summary_ldg_v2_training.png" width="65%"></p>
</p>

### Dataset reconstruction 

<p align="center">
<img src="images/summary_ldg_v2_reconstruction.png" width="85%"></p>
</p>

### Generating new letters/digits

<p align="center">
<img src="images/summary_ldg_v2_testing.png" width="100%"></p>
</p>

## LDG Version 1
### [letter_digit_generator_convolutional-CVAE](https://github.com/sungsujaing/letter_digit_generator_VAE/blob/master/letter_digit_generator_v1.ipynb)
Initial convolutional conditional variational autoencoder model.
* label inputs only to decoder
* training/test data reconstructions were satisfactory, but generation of specific string input was somewhat difficult.

---

### [(ARCHIVE) convolutional VAE: initial model building and function testings on F_MNIST](https://github.com/sungsujaing/letter_digit_generator_VAE/blob/master/convolutional%20beta-VAE%20on%20F_MNIST.ipynb)
#### VAE interpolation from image 1 to image 2

<p align="center">
<img src="interpolation_images_fMNIST/summary.png" width="85%"></p>
</p>

### [(ARCHIVE) Dog_crossbreed_prediction](https://github.com/sungsujaing/letter_digit_generator_VAE/tree/master/(archive)vae_test_standford_dog_breed_dataset/dog_crossbreed_prediction)
While the model architecture seems to be okay, the [standford dogs datasets](http://vision.stanford.edu/aditya86/ImageNetDogs/) may not be suitable to train VAE.
