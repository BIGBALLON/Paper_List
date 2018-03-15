# My Paper Reading List

## Convolutional Neural Network

- (**LeNet**) LeCun, Yann, et al. "**Gradient-based learning applied to document recognition**." Proceedings of the IEEE 86.11 (**1998**).
- (**AlexNet**) Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "**Imagenet classification with deep convolutional neural networks**." Advances in neural information processing systems. (**2012**).
- (**ZFNet**) Zeiler, Matthew D., and Rob Fergus. "**Visualizing and understanding convolutional networks**." European conference on computer vision. Springer, Cham, (**2014**).
- (**NIN**) Lin, Min, Qiang Chen, and Shuicheng Yan. "**Network in network**." (**2013**). [[arXiv:1312.4400][1]]
- (**VGGNet**) Simonyan, Karen, and Andrew Zisserman. "**Very deep convolutional networks for large-scale image recognition**."(2014). [[arXiv:1409.1556][2]]
- (**GoogLeNet**) Szegedy, Christian, et al. "**Going deeper with convolutions**." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
- (**BN**) Ioffe, Sergey, and Christian Szegedy. "**Batch normalization: Accelerating deep network training by reducing internal covariate shift**." International Conference on Machine Learning. (**2015**). [[arXiv:1502.03167][3]]
- (**ResNet**) He, Kaiming, et al. "**Deep residual learning for image recognition**." Proceedings of the IEEE conference on computer vision and pattern recognition. (**2016**). [[arXiv:1512.03385][4]]  [CVPR 2016 Best Paper] :star:
- (**Pre-active**) He, Kaiming, et al. "**Identity mappings in deep residual networks**." European Conference on Computer Vision. Springer International Publishing. (**2016**). [[arXiv:1603.05027][5]]
- (**Wide ResNet**) Zagoruyko, Sergey, and Nikos Komodakis. "**Wide residual networks**." (**2016**). [[arXiv:1605.07146][6]]
- (**ResNeXt**) Xie, Saining, et al. "**Aggregated residual transformations for deep neural networks**." 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, (**2017**). [[arXiv:1611.05431][7]]
- (**DenseNet**) Huang, Gao, et al. "**Densely connected convolutional networks**." (**2016**). [[arXiv:1608.06993][8]] 
- (**DPN**) Chen, Yunpeng, et al. "**Dual path networks**." Advances in Neural Information Processing Systems. (**2017**). [[arXiv:1707.01629][9]]
- (**SENet**) Hu, Jie, Li Shen, and Gang Sun. "**Squeeze-and-excitation networks**." (**2017**). [[arXiv:1709.01507][10]]
- (**CondenseNet**) Huang, Gao, et al. "**CondenseNet: An Efficient DenseNet using Learned Group Convolutions**." (**2017**). [[arXiv:1711.09224][11]] 

## Optimizers

- Kingma, Diederik P., and Jimmy Ba. "**Adam: A method for stochastic optimization.**" arXiv preprint (2014). [[arXiv:1412.6980][12]]
- Ruder, Sebastian. "**An overview of gradient descent optimization algorithms.**" arXiv preprint (2016). [[arXiv:1609.04747][13]]
- Keskar, Nitish Shirish, and Richard Socher. "**Improving Generalization Performance by Switching from Adam to SGD.**" arXiv preprint (2017). [[arXiv:1712.07628][14]]
- Loshchilov, Ilya, and Frank Hutter. "**SGDR: stochastic gradient descent with restarts**." arXiv preprint (2016). [[arXiv:1608.03983][15]] :star:
- Smith, Leslie N. "**Cyclical learning rates for training neural networks.**" Applications of Computer Vision (WACV), 2017 IEEE Winter Conference on. IEEE, 2017. [[arXiv:1506.01186][16]]
- Gastaldi, Xavier. "**Shake-shake regularization.**" arXiv preprint (**2017**). [[arXiv:1705.07485][23]]

## Generative Adversarial Network

- Goodfellow, Ian, et al. "**Generative adversarial nets**." Advances in neural information processing systems. (**2014**). [[arXiv:1406.2661][17]]
- Mirza, Mehdi, and Simon Osindero. "**Conditional generative adversarial nets**." (**2014**). [[arXiv:1411.1784][18]]
- Radford, Alec, Luke Metz, and Soumith Chintala. "**Unsupervised representation learning with deep convolutional generative adversarial networks**." (**2015**). [[arXiv:1511.06434][19]]
- Reed, Scott, et al. "**Generative adversarial text to image synthesis**." (**2016**). [[arXiv:1605.05396][20]]
- Shrivastava, Ashish, et al. "**Learning from simulated and unsupervised images through adversarial training**."(**2016**). [[arXiv:1612.07828][21]]
- Arjovsky, Martin, Soumith Chintala, and Léon Bottou. "**Wasserstein gan**." (**2017**). [[arXiv:1701.07875][22]]


## (Deep) Reinforcement Learning  

- **Value-based**
    - (**DQN**) Deep Q Network 
        - Mnih, Volodymyr, et al. "**Playing atari with deep reinforcement learning**." (**2013**). [[arXiv:1312.5602][24]] 
        - Mnih, Volodymyr, et al. "**Human-level control through deep reinforcement learning**."(**2015**). [[Nature 518.7540][25]] :star:
    - Other improvements:
        - (**DDQN**) Van Hasselt, Hado, Arthur Guez, and David Silver. "**Deep Reinforcement Learning with Double Q-Learning**." AAAI. **2016**. [[arXiv:1509.06461][26]]
        - Schaul, Tom, et al. "**Prioritized experience replay**."(**2015**). [[arXiv:1511.05952][27]] 
        - Wang, Ziyu, et al. "**Dueling network architectures for deep reinforcement learning**." (**2015**). [[arXiv:1511.06581][28]]  [ICML2016 Best Paper]
- **Actor-Critic**
    - (**DDPG**) Lillicrap, Timothy P., et al. "**Continuous control with deep reinforcement learning**." (**2015**). [[arXiv:1509.02971][29]]
    - (**A3C**) Mnih, Volodymyr, et al. "**Asynchronous methods for deep reinforcement learning**." ICML (**2016**). [[arXiv:1602.01783][30]] :star:
    - (**ACER**) Wang, Ziyu, et al. "**Sample efficient actor-critic with experience replay**." (**2016**). [[arXiv:1611.01224][31]]
    - (**ACKTR**) Wu, Yuhuai, et al. "**Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation**." Advances in Neural Information Processing Systems. (**2017**). [[arXiv:1708.05144][32]]
- **More**
    - (**UNREAL**) Jaderberg, Max, et al. "**Reinforcement learning with unsupervised auxiliary tasks**." (**2016**). [[arXiv:1611.05397][33]] 
    - (**TRPO**) Schulman, John, et al. "**Trust region policy optimization**." Proceedings of the 32nd International Conference on Machine Learning (ICML-15). (**2015**). [[arXiv:1502.05477][34]]
    - (**DPPO**) Schulman, John, et al. "**Proximal policy optimization algorithms**." (**2017**). [[arXiv:1707.06347][35]]
    - Heess, Nicolas, et al. "**Emergence of locomotion behaviours in rich environments**." (**2017**). [[arXiv:1707.02286][36]]
    - Hessel, Matteo, et al. "**Rainbow: Combining Improvements in Deep Reinforcement Learning**." (**2017**). [[arXiv:1710.02298][37]]
    - Andrychowicz, Marcin, et al. "**Learning to learn by gradient descent by gradient descent**." Advances in Neural Information Processing Systems. (**2016**). [[arXiv:1606.04474][38]]
    - (**GAIL**)Ho, Jonathan, and Stefano Ermon. "**Generative adversarial imitation learning**." Advances in Neural Information Processing Systems. (**2016**). [[arXiv:1606.03476][39]]
    - (**InfoGAIL**)Li, Yunzhu, Jiaming Song, and Stefano Ermon. "**InfoGAIL: Interpretable Imitation Learning from Visual Demonstrations**." Advances in Neural Information Processing Systems. (**2017**).  [[arXiv:1703.08840][40]]
    - Lample, Guillaume, and Devendra Singh Chaplot. "**Playing FPS Games with Deep Reinforcement Learning**." AAAI. (**2017**). [[arXiv:1609.05521][41]]
    - O'Donoghue, Brendan, et al. "**Combining policy gradient and Q-learning**." (**2016**).  [[arXiv:1611.01626][42]]
    - Merel, Josh, et al. "**Learning human behaviors from motion capture by adversarial imitation**."  (**2017**). [[arXiv:1707.02201][43]]
    - Liu, YuXuan, et al. "**Imitation from observation: Learning to imitate behaviors from raw video via context translation**." (**2017**). [[arXiv:1707.03374][44]]


## Computer Games 

-  **2048 Like Games**
    - Szubert, Marcin, and Wojciech Jaśkowski. "**Temporal difference learning of n-tuple networks for the game 2048**." Computational Intelligence and Games (CIG), IEEE Conference on. IEEE, (**2014**).
    - Wu, I-Chen, et al. "**Multi-stage temporal difference learning for 2048**." Technologies and Applications of Artificial Intelligence. Springer, Cham, (**2014**).
    - Yeh, Kun-Hao, et al. "**Multi-stage temporal difference learning for 2048-like games**." IEEE Transactions on Computational Intelligence and AI in Games (**2016**).
    - Jaskowski, Wojciech. "**Mastering 2048 with Delayed Temporal Coherence Learning, Multi-Stage Weight Promotion, Redundant Encoding and Carousel Shaping**." IEEE Transactions on Computational Intelligence and AI in Games (**2017**). :star:
- **AlphaGo** 
    -  Silver, David, et al. "**Mastering the game of Go with deep neural networks and tree search**." [Nature 529.7587][45] (**2016**): 484-489. :star:
    -  Silver, David, et al. "**Mastering the game of go without human knowledge**." [Nature 550.7676][46] (**2017**): 354. :star:
    -  Silver, David, et al. "**Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm**."  (**2017**). [[arXiv:1712.01815][47]] :star:
- **More**
    - Lai, Matthew. "**Giraffe: Using deep reinforcement learning to play chess**." (**2015**). [arXiv:1509.01549][48] 
    -  Vinyals, Oriol, et al. "**StarCraft II: a new challenge for reinforcement learning**." (**2017**). [[arXiv:1708.04782][49]]    
    -  Maddison, Chris J., et al. "**Move evaluation in go using deep convolutional neural networks**." (**2014**). [[arXiv:1412.6564][50]]
    -  Soeda, Shunsuke, Tomoyuki Kaneko, and Tetsuro Tanaka. "**Dual lambda search and shogi endgames**." Advances in Computer Games. Springer, Berlin, Heidelberg, (**2005**).
    
## Others 

- Li, Yuxi. "**Deep reinforcement learning: An overview**." (**2017**). [[arXiv:1701.07274][51]]
- [AdversarialNetsPapers][52]
- [deep-reinforcement-learning-papers][53]
- [BIGBALLON/cifar-10-cnn][54]
- [aymericdamien/TensorFlow-Examples][55]
- [openai/baselines][56]
- [rlcode/reinforcement-learning][57]


  [1]: https://arxiv.org/abs/1312.4400
  [2]: https://arxiv.org/abs/1409.1556
  [3]: https://arxiv.org/abs/1502.03167
  [4]: https://arxiv.org/abs/1512.03385
  [5]: https://arxiv.org/abs/1603.05027
  [6]: https://arxiv.org/abs/1605.07146
  [7]: https://arxiv.org/abs/1611.05431
  [8]: https://arxiv.org/abs/1608.06993
  [9]: https://arxiv.org/abs/1707.01629
  [10]: https://arxiv.org/abs/1709.01507
  [11]: https://arxiv.org/abs/1711.09224
  [12]: https://arxiv.org/abs/1412.6980
  [13]: https://arxiv.org/abs/1609.04747
  [14]: https://arxiv.org/abs/1712.07628
  [15]: https://arxiv.org/abs/1608.03983
  [16]: https://arxiv.org/abs/1506.01186
  [17]: https://arxiv.org/abs/1406.2661
  [18]: https://arxiv.org/abs/1411.1784
  [19]: https://arxiv.org/abs/1511.06434
  [20]: https://arxiv.org/abs/1605.05396
  [21]: https://arxiv.org/abs/1612.07828
  [22]: https://arxiv.org/abs/1701.07875
  [23]: https://arxiv.org/abs/1705.07485
  [24]: https://arxiv.org/abs/1312.5602
  [25]: https://www.nature.com/articles/nature14236
  [26]: https://arxiv.org/abs/1509.06461
  [27]: https://arxiv.org/abs/1511.05952
  [28]: https://arxiv.org/abs/1511.06581
  [29]: https://arxiv.org/abs/1509.02971
  [30]: https://arxiv.org/abs/1602.01783
  [31]: https://arxiv.org/abs/1611.01224
  [32]: https://arxiv.org/abs/1708.05144
  [33]: https://arxiv.org/abs/1611.05397
  [34]: https://arxiv.org/abs/1502.05477
  [35]: https://arxiv.org/abs/1707.06347
  [36]: https://arxiv.org/abs/1707.02286
  [37]: https://arxiv.org/abs/1710.02298
  [38]: https://arxiv.org/abs/1606.04474
  [39]: https://arxiv.org/abs/1606.03476
  [40]: https://arxiv.org/abs/1703.08840
  [41]: https://arxiv.org/abs/1609.05521
  [42]: https://arxiv.org/abs/1611.01626
  [43]: https://arxiv.org/abs/1707.02201
  [44]: https://arxiv.org/abs/1707.03374
  [45]: https://www.nature.com/articles/nature16961
  [46]: https://www.nature.com/articles/nature24270
  [47]: https://arxiv.org/abs/1712.01815
  [48]: https://arxiv.org/abs/1509.01549
  [49]: https://arxiv.org/abs/1708.04782
  [50]: https://arxiv.org/abs/1412.6564
  [51]: https://arxiv.org/abs/1701.07274
  [52]: https://github.com/zhangqianhui/AdversarialNetsPapers
  [53]: https://github.com/junhyukoh/deep-reinforcement-learning-papers
  [54]: https://github.com/BIGBALLON/cifar-10-cnn
  [55]: https://github.com/aymericdamien/TensorFlow-Examples
  [56]: https://github.com/openai/baselines
  [57]: https://github.com/rlcode/reinforcement-learning