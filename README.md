# vemove
Variational EM for the Separation of Time-Varying Convolutive Mixtures


This software contains the matlab code of the Variational EM algorithm for Source Separation of Moving Sources (VEMOVE), from [inria.fr](https://team.inria.fr/perception/research/vemove/)

```
D. Kounades-Bastian, L. Girin, X. Alameda-Pineda, S. Gannot and R. Horaud, 
"A Variational EM Algorithm for the Separation of Time-Varying Convolutive Audio Mixtures," 
IEEE/ACM Transactions on Audio, Speech, and Language Processing, 
vol. 24, no. 8, pp. 1408-1423, Aug. 2016, doi: 10.1109/TASLP.2016.2554286.


D. Kounades-Bastian, L. Girin, X. Alameda-Pineda, S. Gannot and R. Horaud, 
"A variational EM algorithm for the separation of moving sound sources," 
2015 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 
New Paltz, NY, USA, pp. 1-5, doi: 10.1109/WASPAA.2015.7336936.
```

### MATLAB DEMO

In MATLAB run `example.m`

```python
# example.m will first generate a stereo mix with 3 sources (by loading trueSrc1.wav,..) 
# and then call vemove.m to separate that mix. The estimated sources will be written in
# .wav files (estimatedSrc1.wav,..) into the directory ./results/
# vemove.m implements the VEMOVE algorithm.
# 1) Because of the importance of a "good" initialization to EM algorithms in general, 
# an amount of ground-truth information is used (to initialise the NMF parameters) 
# on example.m the amount of ground-truth information is controlled by the variable "snr" 
# within the function corruptInit.m
# 2) There is no ground-truth information used for the room-acoustics 
#  (the mixing matrices A are initialised blindly)
# The functions in the folder ./aux_tools are downloaded from 
#    http://www.irisa.fr/metiss/ozerov/Software/multi_nmf_toolbox.zip
>> example.m
```


### PAPER

**PDF & slides**
 - [journal](https://inria.hal.science/hal-01301762/document)
 - [conference](https://inria.hal.science/hal-01169764v2/document)
 - [Slides](https://team.inria.fr/perception/files/2016/04/slides_vemove_waspaa2015.pdf)
 - [Variational Kalaman Equations](https://team.inria.fr/perception/files/2017/08/derivationForwardBackwardNEW.pdf)












