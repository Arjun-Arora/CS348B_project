# Monte Carlo Denoising with Deep Learning
### Arjun Arora & Eric Xu

## Introduction
Monte Carlo Integration is the standard technique used in ray tracing to produce physically accurate lighting calculations. While there are a variety of sampling techniques to deal with the variance resulting from an inherently stochastic process (stratified sampling, importance sampling, etc), one still needs on the order of 10<sup>3</sup> to 10<sup>6</sup> rays per pixel (depending on scene complexity) to produce perfectly lit scenes. However, as shown in lecture, increasing ray counts does not scale linearly with perceptual increase in quality and we hit a regime of “diminishing returns” as we increase ray counts from 10<sup>2</sup> to 10<sup>N</sup>. Instead of tracing these many rays per image, we propose the use of a deep neural network to accelerate image space denoising with features generated from the pbrt framework to produce high quality images from noisy low ray count inputs.

## Related Work
For our models, we generally followed the model outlined by Chaitanya et al<sup>1</sup>, with some slight modifications. In the paper, the model used is a Unet with recurrent connections to allow the model to learn temporal stability. This would allow their model to be used as a denoiser for interactive and real time applications. However, since our focus was narrowed to just denoising with no interactive use case, we decided to omit the recurrent component and instead focus on denoising our model using the UNet architecture presented. We also wanted to try a variant on the UNet architecture outlined by Liu, Pengju, et al<sup>2</sup>. This architecture is similar to the UNet except that instead of using normal bilinear downsampling to separate each stage of the expanding and contracting networks, this model uses a harr wavelet transform to "downsample" and "upsample" images from each stage of the networks. In theory, this allows the model to preserve more frequency information than bi-linear downsampling while also providing a perfect reconstruction filter in the form of the inverse discrete wavelet transform. 

## Features
In addition to the raw pixel values for the noisy images, we include three auxiliary features that we concatenate to form as input for training the model. These features are extracted from PBRT during the rendering process. 

* **Shading normals**: Shading normals provide us information about the geometric structures and shapes in the scene. For each ray we shoot, we compute the normal vector at the intersection between that ray and the relevant object. These normals are stored in the SurfaceInteraction object during ray tracing, and we transform these vectors into camera space and take the average over all rays per pixel. 

<img src="http://graphics.stanford.edu/courses/cs348b-19-spring-content/article_images/226_.jpg" width="80%">

* **Depth map**: For each pixel, we also compute the average depth of the intersection points between the ray and the object. Here, depth refers to the z-coordinate of the intersection point in camera space. As a result, we compute the depth map by extracting the intersection point and transforming it into camera space before saving the z-coordinate. 

<img src="http://graphics.stanford.edu/courses/cs348b-19-spring-content/article_images/226_1.jpg" width="80%">

* **Albedo**: Albedo is the proportion of the incident light reflected by a surface, and is arguably the most important feature for denoising images. We obtain the albedo for each pixel by examining the BSDF object in the SurfaceInteraction. We then calculate the albedo by sampling in ten random coordinates in [0, 1]<sup>2</sup> and using monte carlo integration to estimate the rho value.

<img src="http://graphics.stanford.edu/courses/cs348b-19-spring-content/article_images/226_2.jpg" width="80%">

Overall, the we have six additional channels of features for training (two shading normal channels, one depth channel, and three albedo channels). Including the original raw pixels of the noisy image, our final input shape for training is (height of image, width of image, 9 channels). 

## Preprocessing Step
Before training, we perform a couple of preprocessing steps to help the model learn. First, PBRT saves the rendered images as .exr files which are in high dynamic range (HDR) space, but models generally learn better when the pixel values are in low dynamic range (LDR) space. As a result, we tone map the noisy image, reference image, and albedo features by raising the values to the power of 0.2. This tone map transformation is also utilized in Chaitanya et al<sup>1</sup>. Furthermore, we divide the noisy image pixels by the albedo values to obtain the effective irradiance. Thus in order to generate the final output, we multiply the resulting model logits by the albedo. 

## Image Generation and Dataset
We use the contemporary-bathroom and villa-daylight scenes from pbrt-v3-scenes<sup>3</sup> as our training and validation set. To generate the images, we write a script that perturbs the camera position and angle to capture various shots of the scene. The noisy images are generated with 64 samples per pixel and the target reference images are generated with 4096 samples per pixel. 

<img src="http://graphics.stanford.edu/courses/cs348b-19-spring-content/article_images/226_9.jpg" width="80%">

In total, we obtain 150 training images and 37 validation images (87 scenes from contemporary-bathroom and 100 scenes from villa-daylight). We further augment our dataset by sampling eight 128 x 128 patches from each image. This gives us 1200 training patches and 296 validation patches in all.
Finally, we test our denoiser on the sportscar scene (also drawn from pbrt-v3-scenes<sup>3</sup>), which has a completely different distribution than the training and validation inputs. 

<img src="http://graphics.stanford.edu/courses/cs348b-19-spring-content/article_images/226_10.jpg" width="80%">

## Approach
### UNet + Multiwavelet model

<img src = "http://graphics.stanford.edu/courses/cs348b-19-spring-content/article_images/226_14.jpg" width="80%">

* The Unet model we used has five stages for both the expanding and contracting network. Each stage has two 2-dimensional convolution layers followed by a ReLU activation. Following these convolutions is a 2x downsampling layer or 2x upsampling layer for the contracting and expanding network respectively. Each convolution has a 3x3 spatial kernel with a stride of 1 and a padding of 1. The last expanding and first contracting stage both use 32 filters per convolution layer with each successively deeper stage using 32 more filters than the previous layer. This means that with the 5 stage design we have, our final layer had 160 filters.

* The multiwavelet model follows quite closely to the above unet model with one major difference in the down and upsampling layers. The downsampling layers are replaced with a discrete wavelet transform. The DWT creates 4 output transforms corresponding to 1 low pass image, 2 bandpass, and 1 high pass image, all half the size of the input spatial dimensions. These images are then stacked along the channel dimensions and passed through the convolutions of the next contracting stage. The upsampling layers are also in turn replaced with the inverse discrete wavelet transform. This layer produces an output that has one-fourth the channel dimension but 2x the spatial dimensions. Each output of this inverse filter is then passed to the successive expanding stage.

### SmoothL1Loss
We also experimented with using a Smooth L1 Loss rather than the standard L1 loss used in Chaitanya et al<sup>1</sup>. This loss follows the following formula: 

<img src="http://graphics.stanford.edu/courses/cs348b-19-spring-content/article_images/226_4.jpg" width="20%">

<img src="http://graphics.stanford.edu/courses/cs348b-19-spring-content/article_images/226_5.jpg" width="30%">

This loss function essentially acts as L1 loss as long as pixel loss remains above 1 and essentially less sensitive to outliers than L2 loss.
However, as loss per pixel goes below 1, the loss function changes to an L2 loss metric to less heavily penalize outliers and focus on how the overall image looks. We theorized this would also help improve generalizability as loss got very small; however, we found normal L1 loss to work better empirically. 

## Results
We train both the UNet model and the Multiwavelet model for 500 epochs with a batch size of 64. Both models begin with a learning rate of 0.001 and operate under a scheduler that decreases the learning rate by a factor of 10 every ten epochs of stagnant validation loss (threshold for significant change = 1e-4). In total, the training time for the UNet model is 176 minutes while the training time for the Multiwavelet model is 158 minutes. 

The table below records relevant metrics for the two models with the best parameters according to the validation set:

|              |  RMSE  |  PSNR  |  SSIM |
|:------------:|:------:|:------:|:-----:|
|     UNet     | 0.0262 | 31.695 | 0.804 |
| Multiwavelet | 0.0367 | 28.722 | 0.760 |

As we see above, the UNet model performs better than the Multiwavelet model across all three metrics. Therefore, we proceed with the UNet model as the primary denoiser for the rest of the analysis. Below are training and validation plots of L1 loss and the three metrics across the epochs:
<img src="http://graphics.stanford.edu/courses/cs348b-19-spring-content/article_images/226_6.jpg" width="80%">

Overall, we found the the execution time for actually denoising an image with the UNet is around 0.1 seconds, and this scales linearly with the number of pixels in the image. For the contemporary-bathroom scene (which is part of the training and validation distribution), we obtain the following denoised image:

<img src="http://graphics.stanford.edu/courses/cs348b-19-spring-content/article_images/226_7.jpg" width="80%">

The image has a RMSE of 0.0279, PSNR of 31.07, and SSIM of 0.931. This bathroom scene shows the ability of the model to perform well on data within the train/validation distribution of scenes. Here we see that most of the texture detail is preserved, surprisingly even in the reflection of the mirror (though there is some noise). We also see that most of the medium and high frequency noise has been removed completely, leaving only a couple areas of low frequency splotches left, particularly around the cords of the lights. Overall, this result shows the validity of UNet architectures for denoising monte carlo integrated images.

Finally, running the model on the sportscar test image, we achieve the following denoised image:

<img src="http://graphics.stanford.edu/courses/cs348b-19-spring-content/article_images/226_11.jpg" width="80%">

The image has a RMSE of 0.0259, PSNR of 31.736, and SSIM of 0.938. Because this image is outside the distribution of images that make up our training and validation set, the results are qualitatively worse than the bathroom image, even if the evaluation metrics seem close to the bathroom results. While we see most of the noise is gone, the reconstruction results on the highlights of the car leave some room to be desired. The highlights on the hood of the car seemed to be somewhat brown, clearly not matching the red color of the car. As well, there is some medium frequency noise on the door of the car as well as some on the wheels. This indicates that while our model performs very well within the distribution of the villa and bathroom scenes, it may have overfit slightly to that distribution. Moving forward, having a larger sample of scenes and perhaps some more regularization procedures would produce better results on a test image.

<img src="http://graphics.stanford.edu/courses/cs348b-19-spring-content/article_images/226_12.jpg" width="80%">

As a comparison, we also denoised the 64 sample sportscar using Intel Open Image Denoise, and it received an RMSE of 0.00936, PSNR of 40.576, and SSIM of 0.920. Below shows the reconstructed image for reference. 

<img src="http://graphics.stanford.edu/courses/cs348b-19-spring-content/article_images/226_13.jpg" width="80%">

## Distribution of Work
We worked together on the PBRT code to generate and extract the features from the noisy image during rendering. Eric handled the data collection by writing scripts to generate different shots of the scene, and Arjun coded the model architecture and the training pipeline. Both of us worked together on running the final model and producing the denoised output. 

## Problems Encountered
The majority of the problems we faced were in extracting the features we needed from pbrt as well as fine-tuning our model to facilitate training. 

Initially we had some difficulty determining how to extract the bsdf as well as the image space normals. To do so, we had to hack the pbrt integrator Li function to return the surface intersection so we could extract that information ourselves. Moreover, we had to add on the surface Intersection class to also have a depth parameter so we could save z depth. We also had issues when averaging various different features like albedo and depth across the input samples. Initially, we weren’t handling the cases where a ray did not intersect with a scene because we assumed most rays would have to intersect our scene as most of them were indoors. However, there were cases where we didn’t intersect with the scene and with default values for some of our features (like albedo = Spectrum(0.0) and depth = 0), we ended up systematically underestimating albedo and depth.

Our models also gave us trouble in the beginning. Initially our models refused to train and produced very mediocre outputs even after 100 or so epochs. Brennan was a great help here in debugging our training and model scripts. One of our major problems was that because we were extracting our features raw from pbrt, they were all in high dynamic range space, which is very difficult for learning algorithms to learn on with L1 loss. Moreover, our initial architecture included batchnorm layers which have been shown to not work well in the monte carlo denoising space. This is due to the small batch size normally used, meaning the statistics calculated per batch are generally very noisy estimates of the training distribution and can harm training. Our model architectures were also overparameterized with too many parameters in the lower stages of the model rather than the upper stages. By balancing and reducing the filters per dimension we were able to produce even better results. Finally, we were initially training with 64 x 64 patches which, for training images of 1000 x 1000 or larger, are too small of a spatial extent to learn the texture detail of most scenes. By increasing our patch size to 128 x 128 we were able to incorporate a larger spatial extent into our model, drastically improving texture reconstruction. 

Link to code: https://github.com/Arjun-Arora/CS348B_project

### References
1. Chaitanya, Chakravarty R. Alla, et al. "Interactive reconstruction of Monte Carlo image sequences using a recurrent denoising autoencoder." ACM Transactions on Graphics (TOG) 36.4 (2017): 98.

2. Liu, Pengju, et al. "Multi-level wavelet-CNN for image restoration." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2018.

3. Scenes for pbrt-v3: https://pbrt.org/scenes-v3.html
