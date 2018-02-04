## Very Deep Convolutional Networks for Natural Language Processing in Tensorflow 

This is the DenseNet implementation of the paper **Do Convolutional Networks need to be Deep for Text Classification ?** in Tensorflow. We study in the paper the importance of depth in convolutional models for text classification, either when character or word inputs are considered. We show on 5 standard text classification and sentiment analysis tasks that deep models indeed give better performances than shallow networks when the text input is represented as a sequence of characters. However, a simple shallow-and-wide network outperforms deep models such as DenseNet with word inputs. Our shallow word model further establishes new state-of-the-art performances on two datasets: Yelp Binary (95.9\%) and Yelp Full (64.9\%). 

**Paper:**

Hoa T. Le, Christophe Cerisara, Alexandre Denis. **Do Convolutional Networks need to be Deep for Text Classification ?**. Association for the Advancement of Artificial Intelligence 2018 (**AAAI-18**) Workshop on Affective Content Analysis. (https://arxiv.org/abs/1707.04108)

    @article{DBLP:journals/corr/LeCD17,
      author    = {Hoa T. Le and
                   Christophe Cerisara and
                   Alexandre Denis},               
      title     = {Do Convolutional Networks need to be Deep for Text Classification ?},  
      journal   = {CoRR},  
      year      = {2017}  
    }

<p align="center">
  <img src="https://github.com/lethienhoa/Very-Deep-Convolutional-Networks-for-Natural-Language-Processing/blob/master/Selection_134.png?raw=true" />
</p>

**Results:**

<p align="center">
  <img src="https://github.com/lethienhoa/Very-Deep-Convolutional-Networks-for-Natural-Language-Processing/blob/master/Selection_135.png?raw=true" />
</p>

<p align="center">
  <img src="https://github.com/lethienhoa/Very-Deep-Convolutional-Networks-for-Natural-Language-Processing/blob/master/Selection_136.png?raw=true" />
</p>

## Reference Source Codes

- https://github.com/dennybritz/cnn-text-classification-tf

