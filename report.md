# Report 
## DataPreProcessing
* ImageFiltering on Depth Image 
    * [Source about the following explaination](http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_morphology.html)
    * Convert depthmap with all none zero into one and apply follwoing
    * MorphologyEx Open for remove bright unconnected spot 
    * MorphologyEx Close for remove  dark unconnected spot
    * bilaterFilter on mask for denoising 
    * MorphologyEx Grad for segmentation and smoothen edges (aka closest pixles).
    * Compute Image Differences with  `Histogram Comparison` 
        *  Standard Deivation comparison would lead to confusion when image chunk is too small or too large since is 0 and 1
            * code
            ```python 
                _, std = cv2.meanStdDev(prev_label)
                _, std2 = cv2.meanStdDev(tmp)
                diff = np.abs(std2-std)
            ```
        *  Mean square is the same as previous one 
            * code 
            ```python 
            err = np.sum((prev_label.astype(np.float64) - labels.astype(np.float64)) **2 )
            err /=float(prev_label.shape[0] * prev_label.shape[1] )
            ```
        * [Histograms compare based on correlation of the images](https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html)
            * code
            ```python 
            prev_hist, next_hist = cv2.calcHist(
                [prev_label], [0], None, [256], [
                    0, 256]), cv2.calcHist(
                [tmp], [0], None, [256], [
                    0, 256])
            d = cv2.compareHist(prev_hist, next_hist, cv2.HISTCMP_CORREL)
            ```
        * Then apply the most similar filter to the rgb so we obtain the segmented images

### Data Clean Up Normalization
* resize image from `(320,240 ,3)` to `(256 , 256 ,3)` with cv2.resize 
* Normalization data from 0 to 255 into 0 to 1 in float by 
```python 
    X = (X - min(X))/(max(X)-min(X)) # for context min is 0 and max is 255 
```
* Background Color Influence a lot on the result so two kind of background Color sample are also included.
    * black best performance loss value of `0.0110`
    * white  best performance loss value of `0.0315`
    * Probably due to the way i normalizes the data

1. Network Architecture
* Widly insipre by [this](A. Radford, L. Metz, and S. Chintala. Unsupervised representation\nlearning with deep convolutional generative adversarial \n networks. arXiv preprint arXiv:1511.06434, 2015.)
* Encoder just see the picture 
* Laten Variable 
* ![figures NUMBERS](./figures/Encoder.png)
    * InputImageShape (256,256,3)  
    * |Conv2D |FilterSize 64 |KernelFilterSize (3x3)|outputImageShape (128,128,64)
        * |relu |outputImageShape (128,128,64)
    
    * |Conv2D |FilterSize 64|KernelFilterSize (3x3)|activate Relu|outputImageShape (128,128,64)
        * |BatchNormalization |outputImageShape (128,128,64)
        * |relu |outputImageShape (128,128,64)
    
    * |Conv2D |FilterSize 128|KernelFilterSize (3x3)|activate Relu|outputImageShape (64,64,128)
        * |BatchNormalization |outputImageShape (64,64,128)
        * |relu |outputImageShape (64,64,128)
    
    * |Conv2D |FilterSize 256|KernelFilterSize (3x3)|activate Relu|outputImageShape (32,32,256)
        * |BatchNormalization |outputImageShape (32,32,256)
        * |relu |outputImageShape (32,32,256)
    
    * |Conv2D |FilterSize 512|KernelFilterSize (3x3)|activate Relu|outputImageShape (16,16,512)
        * |BatchNormalization |outputImageShape (16,16,512)
        * |relu |outputImageShape (16,16,512)
    
    * |Conv2D |FilterSize 512|KernelFilterSize (3x3)|activate Relu|outputImageShape (8,8,512)
        * |BatchNormalization |outputImageShape (8,8,512)
        * |relu |outputImageShape (8,8,512)
    
    * |Conv2D |FilterSize 512|KernelFilterSize (3x3)|activate Relu|outputImageShape (4,4,512)
        * |BatchNormalization |outputImageShape (4,4,512)
        * |relu |outputImageShape (4,4,512)
    
    * |Conv2D |FilterSize 512|KernelFilterSize (3x3)|activate Relu|outputImageShape (2,2,512)
        * |BatchNormalization |outputImageShape (2,2,512)
        * |relu |outputImageShape (2,2,512)
    
    * |Conv2D |FilterSize 512|KernelFilterSize (3x3)|activate Relu|outputImageShape (1,1,512)
        * |BatchNormalization |outputImageShape (1,1,512)

* Decoder just see the picture

    * |Deconv2D |FilterSize 512|KernelFilterSize (3x3)|activate Relu|outputImageShape (2,2,512)
        * |BatchNormalization |outputImageShape (2,2,512)
        * |Dropout|.5|OutputImageShape (2,2,512)

    * |Deconv2D |FilterSize 512|KernelFilterSize (3x3)|activate Relu|outputImageShape (4,4,512)
        * |BatchNormalization |outputImageShape (4,4,512)
        * |Dropout|.5|OutputImageShape (4,4,512)
    
    * |Deconv2D |FilterSize 512|KernelFilterSize (3x3)|activate Relu|outputImageShape (8,8,512)
        * |BatchNormalization |outputImageShape (8,8,512)
        * |Dropout|.5|OutputImageShape (8,8,512)
    
    * |Deconv2D |FilterSize 512|KernelFilterSize (3x3)|activate Relu|outputImageShape (8,8,512)
        * |BatchNormalization |outputImageShape (8,8,512)
        * |Dropout|.5|OutputImageShape (8,8,512)
    
    * |Deconv2D |FilterSize 512|KernelFilterSize (3x3)|activate Relu|outputImageShape (16,16,512)
        * |BatchNormalization |outputImageShape (16,16,512)
        * |Dropout|.5|OutputImageShape (16,16,512)

    * |Deconv2D |FilterSize 512|KernelFilterSize (3x3)|activate Relu|outputImageShape (32,32,256)
        * |BatchNormalization |outputImageShape (32,32,256)
        * |Dropout|.5|OutputImageShape (32,32,256)
    
    * |Deconv2D |FilterSize 512|KernelFilterSize (3x3)|activate Relu|outputImageShape (64,64,128)
        * |BatchNormalization |outputImageShape (64,64,128)
        * |Dropout|.5|OutputImageShape (64,64,128)
    
    * |Deconv2D |FilterSize 512|KernelFilterSize (3x3)|activate Relu|outputImageShape (128,128,64)
        * |BatchNormalization |outputImageShape (128,128,64)
        * |Dropout|.5|OutputImageShape (128,128,64)

    * |Deconv2D |FilterSize 512|KernelFilterSize (3x3)|activate Relu|outputImageShape (256,256,3)
        * |BatchNormalization |outputImageShape (256,256,3)
        * |Dropout|.5|OutputImageShape (256,256,3)
    * Activation Relu | outputImageShape (256,256,3)

* ![figures NUMBERS](./figures/Decoder.png)

4. Training parameters 
### Fine Tunning in network 
* Learning rate is `1.e-5`
* Training Opts is `Adam(lr=1.e-5 , beta_1 = .9 , beta_2 = .999 , epsion=10e-10)`
* Loss function `l1_loss`
* `batch-size` 200 
* 10 batch per epochs
* black bkground epochs `6145` pick this one 
* White bkground epochs `5615`
* validationSize is  20% of total test sample, test is 10% total of 5214 images
* 70% training|20% validation|10% testing, data is splitted in unifrom random sampling
    * validation Size is 1042 of images  
    * training Size is 522 of images
    * training Size is 3650 of images

