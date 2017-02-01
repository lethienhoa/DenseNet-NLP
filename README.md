## Implementation

This is the implementation of the paper **Very Deep Convolutional Networks for Natural Language Processing** of **A. Conneau et al (2016)** in Tensorflow. This code doesn't employ shorcut because the best performance is observed in 29 layers without shortcut. It's free to choose the embedding size of vector so here it is initialized by one-hot-vector of alphabet's dictionary. The Batch-Normalisation followe the instruction given [here](http://r2rt.com/implementing-batch-normalization-in-tensorflow.html). The model is evaluated on Twitter data set [4].

<p align="center">
  <img src="https://lh6.googleusercontent.com/bp61G9vYu2KjotruD1IFUd8TyZC1VL2BS-Uial0U3zNMvKVYh00tyjg_4fTAzI_NayoqOyZHce6ce_4=w1301-h641" />
</p>

## Reference Articles

- [1] Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015)
- [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
- [3] Alexis Conneau, Holger Schwenk, Loïc Barrault, Yann LeCun. Very Deep Convolutional Networks for Natural Language Processing. CoRR 2016
- [4] Alec Go. Richa Bhayani. Lei Huang. Twitter Sentiment Classification using Distant Supervision. Stanford

## Reference Source Codes

- https://github.com/dennybritz/cnn-text-classification-tf
- http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
- https://github.com/scharmchi/char-level-cnn-tf
- https://github.com/tensorflow/models/tree/master/resnet

--------------
MIT License
