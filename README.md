## Implementation

This model is a CNN based only on the character level as charCNN [1]. It exploit the benefit of deep structure for text understanding task as [3]. The first work of transfer a shallow CNN & distant supervision can be read at [5]. The aim of this model is to transfer a very deep (rather than a shallow) network and base only on character level without any feature-engineering at the word level [5]. This kind of transfer is depicted in [7]:

<p align="center">
  <img src="https://drive.google.com/open?id=0Bx6GxfkLCIEoeFltSFdzX0pDTVU" />
</p>

The code is rewritten under Tensorflow TF-Slim to be shorter and facilitate more easily the tranfer of deep networks' layers, as Inception-v3 [4]. 

The evaluation is on small data set of Semval-2013 (2.3k/0.9k pos/neg tweets). The input is a big Twitter's emoticon data set (1.6 M balanced pos/neg tweets) [6].

## Reference Articles

- [1] Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015)
- [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
- [3] Alexis Conneau, Holger Schwenk, Loïc Barrault, Yann LeCun. Very Deep Convolutional Networks for Natural Language Processing. CoRR 2016

## Reference Source Codes

- https://github.com/dennybritz/cnn-text-classification-tf
- https://github.com/scharmchi/char-level-cnn-tf
- https://github.com/tensorflow/models/tree/master/resnet

