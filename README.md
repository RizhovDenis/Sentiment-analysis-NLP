# Sentiment analysis NLP
The project on segmentation group of people who made possitive or negative comment. I cleared data for training neural networks model.
## Table of contents
- [Tech modules](#tech-modules)
- [Data set description](#data-set-description)
- [Structure neural networks](#structure-neural-networks)
- [Conclusion](#conclusion)
- [Sources](#sources)
## Tech modules
- [tensorflow](https://www.tensorflow.org/)
- [nltk](https://www.nltk.org/)
- [Pandas](https://pandas.pydata.org/)
- [sklearn](https://scikit-learn.org/stable/)
- [Numpy](https://numpy.org/)
## Data set description
I took data set on [kaggle competition](https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview/description).
## Structure neural networks
<p align="center">
    <h3 align="center">Long short-term memory</h3>
</p>
<p align="center">
    <img src="./assets/lstm.png" />
</p>

The compact forms of the equations for the forward pass of an LSTM cell with a forget gate are:

<div align="left"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=f_t%20%3D%20%5Csigma_g(W_%7Bf%7D%20x_t%20%2B%20U_%7Bf%7D%20h_%7Bt-1%7D%20%2B%20b_f)%20%5C%5C"></div>
<div align="left"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=i_t%20%3D%20%5Csigma_g(W_%7Bi%7D%20x_t%20%2B%20U_%7Bi%7D%20h_%7Bt-1%7D%20%2B%20b_i)%20%5C%5C"></div>
<div align="left"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=o_t%20%3D%20%5Csigma_g(W_%7Bo%7D%20x_t%20%2B%20U_%7Bo%7D%20h_%7Bt-1%7D%20%2B%20b_o)%20%5C%5C"></div>
<div align="left"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Ctilde%7Bc%7D_t%20%3D%20%5Csigma_c(W_%7Bc%7D%20x_t%20%2B%20U_%7Bc%7D%20h_%7Bt-1%7D%20%2B%20b_c)%20%5C%5C"></div>
<div align="left"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=c_t%20%3D%20f_t%20%5Ccirc%20c_%7Bt-1%7D%20%2B%20i_t%20%5Ccirc%20%5Ctilde%7Bc%7D_t%20%5C%5C"></div>
<div align="left"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=h_t%20%3D%20o_t%20%5Ccirc%20%5Csigma_h(c_t)"></div>

#### Variables

* <div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=x_t%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%7D"></div>
input vector to the LSTM unit
* <div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=f_t%20%5Cin%20%7B(0%2C1)%7D%5E%7Bh%7D"></div>
forget gate's activation vector
* <div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=i_t%20%5Cin%20%7B(0%2C1)%7D%5E%7Bh%7D"></div> input/update gate's activation vector
* <div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=o_t%20%5Cin%20%7B(0%2C1)%7D%5E%7Bh%7D"></div>
output gate's activation vector
* <div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=h_t%20%5Cin%20%7B(-1%2C1)%7D%5E%7Bh%7D%20"></div> hidden state vector also known as output vector of the LSTM unit
* <div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Ctilde%7Bc%7D_t%20%5Cin%20%7B(-1%2C1)%7D%5E%7Bh%7D"></div> 
cell input activation vector
* <div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=c_t%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bh%7D"></div>
cell state vector
* <div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=W%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bh%20%5Ctimes%20d%7D%2C%20U%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bh%20%5Ctimes%20h%7D%5Ctextnormal%7B%20%7D%20and%20%5Ctextnormal%7B%20%7D%20b%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bh%7D"></div> 
weight matrices and bias vector parameters which need to be learned during training

#### Activation functions
* <div align="left"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Csigma_g%3A%20sigmoid%20function."></div>
* <div align="left"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Csigma_c%3A%20hyperbolic%20tangent%20function."></div>
* <div align="left"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Csigma_h%3A%20hyperbolic%20tangent%20function%2C%20or%20as%20the%20peephole%20LSTM%20paper%20suggests%2C"></div>
<div align="left"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Csigma_h(x)%20%3D%20x"></div>

# Conclusion
On test data accuracy is 84.45%. Hypotheses of improving results: Increase train data set; Upgrade neural network; Work with lemmatization.
# Sources
- [wikipedia.org](https://en.wikipedia.org/wiki/Long_short-term_memory)
