# 100DaysOfMLCode
Repository of the AI journey 100 Days of ML Code


# 100 Days Of ML Code - Log

## Day 1 : October 9th 2018
1 Hour

Introduction into Q Learning
Difference between On Policy and Off Policies.

Q Learning is Off Policy Method.
### Temporal differences is similar to a moving average.
### Adaptive learning rate - to assist in speed. If Alpha is too high, we may miss the sweet spot, if its too low, then its to slow to train.

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
[Lecture 4.5 Q-Learning Tutorial of Move 37](https://www.youtube.com/watch?v=tU6_Fc6bKyQ)
[Neural Material Synthesis, This Time On Steroids](https://www.youtube.com/watch?v=UkWnExEFADI&feature=em-uploademail)
[How AlphaGo Zero works - Google DeepMind](https://www.youtube.com/watch?v=MgowR4pq3e8)
[Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8)

*Links*
Daily Reading Material
https://arxiv.org/list/cs.AI/recent
https://arxiv.org/list/cs.LG/recent

# Concepts to understand

Q-Learning

Autoencoders
Temporal Difference
Reinforcement learning
Monte Carlo Tree Search
neural impainting

Disentangled VAE

Outcome of course -
Denoise
Voice recognition - speech to text.
Text to image classification, ie find an image based on text, the text isnt a category but a convnet.
Form a sentence and form various images based on the sentence.

*Need to read*
- Disentangled VAE's (DeepMind 2016): https://arxiv.org/abs/1606.05579
- Applying disentangled VAE's to RL: DARLA (DeepMind 2017): https://arxiv.org/abs/1707.08475
- Original VAE paper (2013): https://arxiv.org/abs/1312.6114
