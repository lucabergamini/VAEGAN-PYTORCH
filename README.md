# VAEGAN-PYTORCH
VAEGAN from "Autoencoding beyond pixels using a learned similarity metric" implemented in Pytorch.
Clean, clear and with comments.

## Requirements
* pytorch
* torchvision
* tensorboard-pytorch
* progressbar2
* matplotlib

For packages version please refer to my ```pip freeze``` command output below:

```
bleach==1.5.0
cycler==0.10.0
decorator==4.1.2
enum34==1.1.6
html5lib==0.9999999
ipython==6.2.1
ipython-genutils==0.2.0
jedi==0.11.0
Markdown==2.6.9
matplotlib==2.0.2
numpy==1.13.3
olefile==0.44
parso==0.1.0
pexpect==4.2.1
pickleshare==0.7.4
Pillow==4.3.0
pkg-resources==0.0.0
progressbar2==3.34.3
prompt-toolkit==1.0.15
protobuf==3.4.0
ptyprocess==0.5.2
Pygments==2.2.0
pyparsing==2.2.0
PyQt5==5.9.1
python-dateutil==2.6.1
python-utils==2.2.0
pytz==2017.3
PyYAML==3.12
scipy==1.0.0
simplegeneric==0.8.1
sip==4.19.5
six==1.11.0
tensorboardX==0.8
tensorflow==1.4.0
tensorflow-tensorboard==0.4.0rc2
torch==0.2.0+28f3d50
torchvision==0.1.9
traitlets==4.3.2
wcwidth==0.1.7
Werkzeug==0.12.2
```
## Visual Results
Results after 13 epochs using lr=0.0001

![Alt text](/results/original.png?raw=true "Original")
![Alt text](/results/recon.png?raw=true "Reconstructed")
![Alt text](/results/sampled.png?raw=true "Sampled")

Reconstructed are not bad (images never seen before), still generated could be better.

## Implementation details
So, using GAN makes training REALLY unstable. In every moment the generator or the descriminator could collapse, rendering awful results. As such, some tricks have been employed in the original implementation (and also here) to try to solve this instability:

### Equilibrium Theory
As one of the two player in the minmax game of the adversarial train tends to overcome the other and to break the equilibrium, the former gets punished by stopping its update. This is achieved thanks to separate optimizers for each of the 3 sub-network of the implementation. The equilibrium value is set from the orginal implementation.

### Gradient Clip
Even if it's not used in this implementation (nor in the original as far as I know), some projects out there clip the gradient from each of the 3 losses between ```[-1,1]```. This could prevent degenerative patterns.

### Low Learning Rate
0.0001 is a really low lr, but even a slightly higher could lead to strange patterns to appear.

### Exponential Decay
Don't know if this helps really :)

## Theory Explanation
Here be Dragons.

### VAE
Plain vae makes just a statistical assumptions: everything is a Gaussian (and that works btw).
The base structure is formed by an encoder and a decoder...so it's an autoencoder? NO my fella, it's not. In fact, What you decode it's not the code you have generated, but a sample from a gaussian space (whose parameters are what you have generated). So you're not encoding a discrete latent space, but the parameters of it.

The final loss is just the reconstruction error between original and reconstructed images, plus a KL-divergence.

#### What the heck is KL-divergence?
To put it simple, is just a way to bring closer two distributions. In our case we want our latent distribution to be a gaussian N(0,I), so we can sample from it using only samples from a standard gaussian N(0,I). If the KLD is not included in the loss function the latent space could be spreaded out in the N-dimensional space, and our samples for the test phase would be just random noise.

### GAN
I really don't have enough time to explain to you why I hate so much GAN. They are unstable, hard to train and to understand, but they do work (sometimes). Moving from a plain VAE to the VAEGAN has been a pain in the ass and it tooks me 2 weeks, so I think i'm not well-suited to talk you about them. What we need to know here is that to obtain the VAEGAN we just stick a discriminator at the end of the plain VAE.

### VAEGAN
Instead of forcing a low reconstruction error, VAEGAN imposes a low errore between intermediate features in the descriminator. If you think about it, if the reconstructed image is very similar to the original one their middle-representation in the descriminator should be similar too. This is why the paper drops the use of what they call "element-wise-error", and prefers the "feature-wise-error"(FWE). They also made some strong assumption on the GAN loss, as they use only the original batch and the sampled from a gaussian N(0,I) to compute it, leaving out the reconstructed batch.

##### Encoder Loss
```KLD+FWE```
So latent space close to a gaussian, but with samples resembling the originals for the descriminator.

##### Decoder Loss
```alpha*FWE-GAN```
Yeah I know...this is not how GAN are usually trained for the generator part, as one should swap the labels (so fake became real and real became fake). I'm still wondering if they lead to the same results (my graphs seems to suggest otherwise).
Alpha here is really low (1e-06 if i remember correctly), probably beacuse the error is computed using the sum fro the single images (so it's the mean of the error between layers, and not the mean of the mean error of layers..what did I just write?)
##### Gan Loss
```GAN```
Nothing special to say here, including the reconstructed loss seems to lower the results (and I REALLY REALLY don't understand why). There are too many hyperparams to investigate them all.



## TODO
- [x] requirements
- [x] visual results
- [ ] TB log
- [x] theory explanation
- [x] implementation details
