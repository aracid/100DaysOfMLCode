# 100DaysOfMLCode
Repository of the AI journey 100 Days of ML Code
Total Time : 4 Hours

# 100 Days Of ML Code

## Day 1 : October 9th 2018
1 Hour

### Introduction into Q Learning
Difference between On Policy and Off Policies.

- Q Learning is Off Policy Method.
- Temporal differences is similar to a moving average.
- Adaptive learning rate - to assist in speed. If Alpha is too high, we may miss the sweet spot, if its too low, then its to slow to train.

### Neural Material Synthesis
Neural Material Synthesis is something that interests me, and as I am currently working on an animated feature that requires synthetic textures, this approach would make sense.

### Alpha go has been revised
  1) Only self play
  2) Predefined features, no hand crafted features.
  3) Change from inception to res-net (Residual)
  4) Combined policy and value network

### Variational Autoencoders
Compress data into a space
AutoEncoder - represent high dimensional data into low dimensional data.
VAE is different because it has 2 terms, mean and standard deviation.


**Links:**

- [x] [Lecture 4.5 Q-Learning Tutorial of Move 37](https://www.youtube.com/watch?v=tU6_Fc6bKyQ)

- [x] [Neural Material Synthesis, This Time On Steroids](https://www.youtube.com/watch?v=UkWnExEFADI&feature=em-uploademail)

- [x] [How AlphaGo Zero works - Google DeepMind](https://www.youtube.com/watch?v=MgowR4pq3e8)

- [x] [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8)


## Day 2 : October 10th 2018
1 Hour

### Proximal Policy Optimization

[Policy Gradient methods and Proximal Policy Optimization (PPO): diving into Deep RL!](https://www.youtube.com/watch?v=5P7I-xPq8u8)
- Learns online.
- Learns directly from the environment.
I didn't quiet follow and will have to re-watch

### Reinforcement Learning
[An introduction to Reinforcement Learning](https://www.youtube.com/watch?v=JgvyzIkgxF0)
The network that transforms input frames to output actions is called the policy network.

### Overcoming sparse rewards in Deep RL: Curiosity, hindsight & auxiliary tasks.
[Overcoming sparse rewards in Deep RL: Curiosity, hindsight & auxiliary tasks.](https://www.youtube.com/watch?v=0Ey02HT_1Ho)


## Day 3 : October 11th 2018
45 Min
[A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8)
The best explanation of Log I've come across.
Entropy : The average amount of information you get from one sample drawn given from the probability distribution P.

[TensorFlow 2.0 Changes](https://www.youtube.com/watch?v=WTNH0tcscqo)

## Day 3 : Octover 12th 2018
4 Hours

[IBM COMMUNITY DAY : Jump Start AI at your Organization](https://www.ibmai-platform.bemyapp.com/#/conference/5bbea3d1bfd6260003e21103)
A wonderful talk by Pam Askar

https://www.ironsidegroup.com/
https://www.ironsidegroup.com/event/intro-ai-workshop/

[IBM COMMUNITY DAY : Model Asset Exchange](https://www.ibmai-platform.bemyapp.com/#/conference/5bb53d8db4ae3f00044cb9f2)

https://developer.ibm.com/code/exchanges/models/

Fabric for Deep Learning

Look at
- https://github.com/IBM/MAX-Fast-Neural-Style-Transfer
- https://github.com/IBM/MAX-Scene-Classifier
- https://kubernetes.io/
- https://developer.ibm.com/patterns/create-a-web-app-to-interact-with-machine-learning-generated-image-captions/

- [x] [Deep RL Bootcamp Lecture 4B Policy Gradients Revisited](https://www.youtube.com/watch?v=tqrcjHuNdmQ)

- [x] [PyTorch Lecture 11: Advanced CNN](https://www.youtube.com/watch?v=hqYfqNAQIjE)

## Day 4 : October 14th 2018

Today I went through a DL course, started playing with Keras Gan tutorial. Looked at VCA and need to learn more about.

- [PyTorch Lecture 07: Wide and Deep](https://www.youtube.com/watch?v=Mf8jna42p2M)

- [x] [GAN by Example using Keras on Tensorflow Backend](https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0)
- [x] [Knowledge Graphs & Deep Learning at YouTube](https://www.youtube.com/watch?v=zzTbptEdKhY)
- [x] [Balancing Recurrent Neural Network sequence data for our crypto predicting RNN - Deep Learning basics with Python, TensorFlow and Keras p.10](https://pythonprogramming.net/balancing-rnn-data-deep-learning-python-tensorflow-keras/)
- [x] [Cryptocurrency-predicting RNN Model - Deep Learning basics with Python, TensorFlow and Keras p.11](https://pythonprogramming.net/crypto-rnn-model-deep-learning-python-tensorflow-keras/)
- [x] [Generative Model Basics (Character-Level) - Unconventional Neural Networks in Python and Tensorflow p.1](https://pythonprogramming.net/generative-model-python-playing-neural-network-tensorflow/)
- [x] [Generating Pythonic code with Character Generative Model - Unconventional Neural Networks in Python and Tensorflow p.2](https://pythonprogramming.net/generating-python-playing-neural-network-tensorflow/)

- [x] [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)

Spent a fair amount of time working through the examples in this tutorial, trained using images our of Redshift.



# *Links*

Daily Reading Material

https://arxiv.org/list/cs.AI/recent

https://arxiv.org/list/cs.LG/recent

# Concepts to understand

- [ ] Trust Region Policy Optimization - John Schulman Berkeley
- [ ] Q-Learning
- [ ] Autoencoders
- [ ] Temporal Difference
- [ ] Reinforcement learning
- [ ] Monte Carlo Tree Search
- [ ] neural impainting
- [ ] Reinforcement Learning
- [ ] Disentangled VAE
- [ ] Transfer learning


- [ ] Image Denoiser
- [ ] Voice recognition - speech to text.
- [ ] Text to image classification, ie. find an image based on text, the text isn't a category but a convnet.
- [ ] Form a sentence and form various images based on the sentence.


*Need to read*
- [ ]Disentangled VAE's (DeepMind 2016): https://arxiv.org/abs/1606.05579
- [ ]Applying disentangled VAE's to RL: DARLA (DeepMind 2017): https://arxiv.org/abs/1707.08475
- [ ]Original VAE paper (2013): https://arxiv.org/abs/1312.6114
- [ ]Reinforcement Learning with Unsupervised Auxiliary Tasks - DeepMind:https://arxiv.org/abs/1611.05397
- [ ]Curiosity Driven Exploration - UC Berkeley: https://arxiv.org/abs/1705.05363
- [ ]Hindsight Experience Replay - OpenAI: https://arxiv.org/abs/1707.01495
- [ ][NumPy Tutorial: Data analysis with Python](https://www.dataquest.io/blog/numpy-tutorial-python/)
- [ ][Deep Neural Networks for YouTube Recommendations](https://ai.google/research/pubs/pub45530)
- [ ][Deep Video Analytics](https://github.com/AKSHAYUBHAT/DeepVideoAnalytics)
- [ ][Deploying Deep Learning](https://github.com/dusty-nv/jetson-inference)
- [ ][ML-Agents Toolkit v0.5, new resources for AI researchers available now](https://blogs.unity3d.com/2018/09/11/ml-agents-toolkit-v0-5-new-resources-for-ai-researchers-available-now/)

*Need to watch*
- [ ] [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures?authuser=0)
  - [x] [1 Intro to MDPs and Exact Solution Methods -- Pieter Abbeel](https://www.youtube.com/watch?v=qaMdN6LS9rA)
  - [x] [Deep RL Bootcamp Lecture 4B Policy Gradients Revisited](https://www.youtube.com/watch?v=tqrcjHuNdmQ)
- [ ] [Easier data analysis in Python with pandas](https://www.dataschool.io/easier-data-analysis-with-pandas/)
- [ ] [Graph neural networks: Variations and applications](https://www.youtube.com/watch?v=cWIeTMklzNg)
- [ ] [Intro to Graph Convolutional Networks](https://www.youtube.com/watch?v=UAwrDY_Bcdc)

# Courses to do
- [ ][INTRODUCTION TO DEEP LEARNING](https://www.nvidia.com/en-us/deep-learning-ai/education/)

# Authors to follow

# Vloggers to follow
- [Arxiv Insights](https://www.youtube.com/channel/UCNIkB2IeJ-6AmZv7bQ1oBYg)

- [Sung Kim](https://www.youtube.com/user/hunkims/videos)
  - [x] [PyTorch Lecture 07: Wide and Deep](https://www.youtube.com/watch?v=Mf8jna42p2M)
  - [x] [PyTorch Lecture 11: Advanced CNN](https://www.youtube.com/watch?v=hqYfqNAQIjE)

  [Sentdex]
- [x][Deep Learning with Python, TensorFlow, and Keras tutorial](https://www.youtube.com/watch?v=wQ8BIBpya2k)
- [ ][Deep Learning in the Browser with TensorFlow.js Tutorial Introduction - TensorFlow.js Tutorial p.1](https://pythonprogramming.net/deep-learning-browser-introduction-tensorflowjs/)
- [x][Balancing Recurrent Neural Network sequence data for our crypto predicting RNN - Deep Learning basics with Python, TensorFlow and Keras p.10](https://pythonprogramming.net/balancing-rnn-data-deep-learning-python-tensorflow-keras/)
- [x][Cryptocurrency-predicting RNN Model - Deep Learning basics with Python, TensorFlow and Keras p.11](https://pythonprogramming.net/crypto-rnn-model-deep-learning-python-tensorflow-keras/)

- [Brandon Rohrer](https://www.youtube.com/user/BrandonRohrer)

# Videos To Watch

- [ ] [Quantum Machine Learning LIVE](https://www.youtube.com/watch?time_continue=553&v=AAO4oq2M_48)

Bloomberg
- [ ] [1. Black Box Machine Learning](https://www.youtube.com/watch?v=MsD28INtSv8)
- [ ] [2. Case Study: Churn Prediction](https://www.youtube.com/watch?v=kE_t3Mm8Z50)
- [ ] [3. Introduction to Statistical Learning Theory](https://www.youtube.com/watch?v=rqJ8SrnmWu0)
- [ ] [4. Stochastic Gradient Descent](https://www.youtube.com/watch?v=5TZww5bTROE)
- [ ] [5. Excess Risk Decomposition](https://www.youtube.com/watch?v=YA_CE9jat4I)
- [ ] [6. L1 & L2 Regularization](https://www.youtube.com/watch?v=d6XDOS4btck)
- [ ] [7. Lasso, Ridge, and Elastic Net](https://www.youtube.com/watch?v=KIoz_aa1ed4)
- [ ] [8. Loss Functions for Regression and Classification](https://www.youtube.com/watch?v=1oi_Mwozj5w)
- [ ] [9. Lagrangian Duality and Convex Optimization](https://www.youtube.com/watch?v=thuYiebq1cE)
- [ ] [10. Support Vector Machines](https://www.youtube.com/watch?v=9zi6-RjlYrU)
- [ ] [11. Subgradient Descent](https://www.youtube.com/watch?v=jYtCiV1aP44)
- [ ] [12. Feature Extraction](https://www.youtube.com/watch?v=gmli6EyiNRw)
- [ ] [13. Kernel Methods](https://www.youtube.com/watch?v=m1otj-SdwYw)
- [ ] [14. Performance Evaluation](https://www.youtube.com/watch?v=xMyAL0C6cPY)
- [ ] [15. "City Sense": Probabilistic Modeling for Unusual Behavior Detection](https://www.youtube.com/watch?v=6nolrvzXiE4)
- [ ] [16. Maximum Likelihood Estimation](https://www.youtube.com/watch?v=ec_5vvxW7fE)
- [ ] [17. Conditional Probability Models](https://www.youtube.com/watch?v=JrFj0xpGd2Q)
- [ ] [18. Bayesian Methods](https://www.youtube.com/watch?v=VCfrGjDPC6k)
- [ ] [19. Bayesian Conditional Probability Models](https://www.youtube.com/watch?v=Mo4p2B37LwY)
- [ ] [20. Classification and Regression Trees](https://www.youtube.com/watch?v=GZuweldJWrM)
- [ ] [21. Basic Statistics and a Bit of Bootstrap](https://www.youtube.com/watch?v=lr5WH-JVT5I)
- [ ] [22. Bagging and Random Forest](https://www.youtube.com/watch?v=f2S4hVs-ESw)
- [ ] [23. Gradient Boosting](hhttps://www.youtube.com/watch?v=fz1H03ZKvLM)
- [ ] [24. Multiclass and Introduction to Structured Prediction](https://www.youtube.com/watch?v=WMQwtoMUjDA)
- [ ] [25. k-Means Clustering](https://www.youtube.com/watch?v=J0A_tkIgutw)
- [ ] [26. Gaussian Mixture Models](https://www.youtube.com/watch?v=I9dfOMAhsug)
- [ ] [27. EM Algorithm for Latent Variable Models](https://www.youtube.com/watch?v=lMShR1vjbUo)
- [ ] [28. Neural Networks](https://www.youtube.com/watch?v=Wr11D5sObzc)
- [ ] [29. Backpropagation and the Chain Rule](https://www.youtube.com/watch?v=XIpyEvLv93A)
- [ ] [30. Next Steps](https://www.youtube.com/watch?v=RMmAVrhAfWs)



# Other Git Repositories

[Awesome-TensorFlow](https://github.com/jtoy/awesome-tensorflow)

[AdversarialNetsPapers](https://github.com/zhangqianhui/AdversarialNetsPapers)

[prakhar21/100-Days-of-ML](https://github.com/prakhar21/100-Days-of-ML)

[Deep Learning: Theory and Experiments](https://github.com/roatienza/Deep-Learning-Experiments)

[Data Science](https://www.youtube.com/user/sgpolitics/videos)

[Papers on deep learning for video analysis](https://github.com/HuaizhengZhang/Papers-on-deep-learning-for-video-analysis)
