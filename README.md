## Implementation of the paper Very Deep Convolutional Networks for Natural Language Processing in Tensorflow 

This is the implementation of the paper **Very Deep Convolutional Networks for Natural Language Processing** of **A. Conneau et al (2016)** in Tensorflow. This code doesn't employ shorcut because the best performance is observed in 29 layers without shortcut. It's free to choose the embedding size of vector so here it is initialized by one-hot-vector of alphabet's dictionary. The model is evaluated on Twitter data set [4].

<p align="center">
  <img src="https://github.com/lethienhoa/Very-Deep-Convolutional-Networks-for-Natural-Language-Processing/blob/master/Selection_042.png?raw=true" />
</p>

## Empirical results

We study in the paper the importance of depth in convolutional models for text classification, either when character or word inputs are considered. We show on 5 standard text classification and sentiment analysis tasks that deep models indeed give better performances than shallow networks when the text input is represented as a sequence of characters. However, a simple shallow-and-wide network outperforms deep models such as DenseNet with word inputs. Our shallow word model further establishes new state-of-the-art performances on two datasets: Yelp Binary (95.9\%) and Yelp Full (64.9\%). 

Hoa T. Le, Christophe Cerisara, Alexandre Denis. **Do Convolutional Networks need to be Deep for Text Classification ?**. Arxiv 2017 (https://arxiv.org/abs/1707.04108)

@article{DBLP:journals/corr/LeCD17,

  author    = {Hoa T. Le and
               Christophe Cerisara and
               Alexandre Denis},
               
  title     = {Do Convolutional Networks need to be Deep for Text Classification ?},
  
  journal   = {CoRR},
  
  year      = {2017}
  
}

## Reference Articles

- [1] Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015)
- [2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
- [3] Alexis Conneau, Holger Schwenk, Loïc Barrault, Yann LeCun. Very Deep Convolutional Networks for Natural Language Processing. CoRR 2016
- [4] Alec Go. Richa Bhayani. Lei Huang. Twitter Sentiment Classification using Distant Supervision. Stanford

## Reference Source Codes

- https://github.com/dennybritz/cnn-text-classification-tf

--------------
MIT License
