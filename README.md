## Implementation

This is the implementation of the paper **Very Deep Convolutional Networks for Natural Language Processing** of **A. Conneau et al (2016)** in Tensorflow. This code doesn't employ shorcut because the best performance is observed in 29 layers without shortcut. It's free to choose the embedding size of vector so here it is initialized by one-hot-vector of alphabet's dictionary. The model is evaluated on Twitter data set [4].

<p align="center">
  <img src="https://github.com/lethienhoa/Very-Deep-Convolutional-Networks-for-Natural-Language-Processing/blob/master/Selection_042.png?raw=true" />
</p>

## Reference Articles

- [1] Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015)
- [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
- [3] Alexis Conneau, Holger Schwenk, Loïc Barrault, Yann LeCun. Very Deep Convolutional Networks for Natural Language Processing. CoRR 2016
- [4] Alec Go. Richa Bhayani. Lei Huang. Twitter Sentiment Classification using Distant Supervision. Stanford

## Reference Source Codes

- https://github.com/dennybritz/cnn-text-classification-tf
- https://github.com/tensorflow/models/tree/master/resnet

--------------
MIT License
