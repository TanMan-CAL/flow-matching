Recently I've been super interested in diffusion and generative models. This repo documents my understanding and time spent on how image generation methods work. Something that really helped me grasp the high dimensional math that is often abstracted is to think of an image as a point in high dimensional space: `(R,G,B) * width (pixels) * height (pixels)`. Random noise is one type of point in this space, and a real image is another type of point. The goal of any image generation model is to learn how to move from the noise region of this space toward the real image region.

A helpful first intuition is Gaussian blur. The Guassian blur is really just understood as the average of the pixels around it. Radius defined as the `nxn` matrix that you take around it. Bigger the blur radius, the more blur the picture is. Original, n = 5, n = 10.

<img src="photos/strawberry.jpg" width="150" /> <img src="photos/image (2).png" width="150" /> <img src="photos/image (5).png" width="150" />
<img src="photos/landscape.jpg" width="150" /> <img src="photos/image (11).png" width="150" /> <img src="photos/image (12).png" width="150" />

Define forward process as moving from true image to noise, and reverse as moving from noise to true image; forward is easier :). Diffusion models learn a reverse process starting from noise to true image as a sequence of transformations. Why can't we do it one step?

All at once from pure noise is impossible to stabilize in one pass, and far too slow and computationally expensive. When the number of noise scales approaches infinity, we essentially perturb the distribution of data with continuously growing levels of noise. Then, the noise perturbation procedure is a **continuous-time stochastic process**. 

Really any generative image process is a stochastic process (super high-level explanation): drift term + diffusion term. The drift term is what pushes generation towards noise during the forward process and towards structure in the reverse process. The diffusion term is some fuction $ g(t) $ for random noise with tiny Gaussian noise steps. This is better explained in Equation 6 & 7 in the paper: https://arxiv.org/pdf/2006.11239.

<img src="photos/image1.png" width="500" />

Solving a reverse SDE yields a score based generative model. Transforming data to a noise distribution can be done with an SDE. It can be reversed to generate samples from noise if we know the score of the distribution at each time step.

Score based generative models (SGMs) are a new paradigm in generation but foundational to diffusion. The score function $ \nabla_{x} \log {p_t}(x) $ is a vector field pointing toward the highest density regions, where data is more likely to exist at a given noise level.

<img src="image.png" width="350" />

Each arrow points in the direction where the probability density increases fastest. At high noise levels, a sample might begin in a low density region, but by repeatedly learning the score field, the reverse process moves it toward higher density regions where realistic images are more likely to exist (green in image).

Quick 1D example (trivial but helpful admist so much abstract math):

<img src="image-2.png" width="350" />

Now think back to what I said above, think of images as points in high-dimensional space (trust me, it makes this stuff so much easier). True images lie on a tiny structured subset of the full space, and much of the space is utter garbage. But training on just the clean image distribution is not feasible as the density is sharp and concentrated.

So we add Gaussian noise to the image, we create a new noisy distribution.
<img src="image-1.png" width="750" />

Instead of learning the clean distribution, we learn the noisy image distribution: $ s(x,t)\approx\nabla_{x} \log {p_t}(x) $. 


The work in this repository is slightly different. Rather than training a score network directly, I used flow matching. For info on score based diffusion: https://yang-song.net/blog/2021/score/.


In flow matching, the model instead learns a velocity field that describes how a sample should move along a continuous path from noise to data.



The main purpose is to improve quality of conditional diffusion models without needing a separate, pre-trained classifier.

The fundamental problem with diffusion (unconditional generation) is that diffusion goes from noise to image step by step. This means it is highly unsuitable to generate a desired output because the generation cannot be controlled. So the question becomes: can you perturb the denoising trajectory?

Answer is classifier guidance generation. First, this means to train a noisy image classifier to generate a class label. Classifier output is $ \nabla_{x_t} \log p(y \mid x_t) $. That is a noise component that is more aligned for a specific class.

What this yields is $ \gamma \, \nabla_{x_t} \log p(y \mid x_t) $ where $ \gamma > 1  $ to amplify the signal. This shifts probability mass from the least likely to the most likely class values, meaning the greater the $ \gamma $, the more class consistent the generation. $ \gamma $ is the inverse temperature parameter. For non-technical people, low $ \gamma $ would mean exploring new options and high $ \gamma $ would mean exploitation of the best option.

So the old $ \epsilon_\theta(x_t, t) $ now is $ \hat{\epsilon}(x_t, t, y) = \epsilon_\theta(x_t, t) - \gamma \, \nabla_{x_t} \log p(y \mid x_t) $; this an overtly high-level explanation of how this approximation is made. More info: https://arxiv.org/pdf/2207.12598.

We basically sample gradients from a classifier when classifying an image of a desired class and feed that gradient to the diffusion model to ultimately perturb the model.

How to get class guidance without an independent classifier? Use the diffusion model itself to get perturbations.

Instead of training a separate classifier, we train a single diffusion model that can operate both conditionally and unconditionally. This is done using conditioning dropout, where the conditioning variable (the thing you want the model to generate according to): $ y $ is randomly removed during training. When $ y $ is removed, the model learns to behave like an unconditional model.

At inference time, we can calculate both $ \epsilon_\theta(x_t, t, y) $ and $ \epsilon_\theta(x_t, t, \phi) $ using the same network. The difference between these implicitly captures the direction toward the class, replacing the need for $ \nabla_{x_t} \log p(y \mid x_t) $.

This leads to classifier-free guidance: $ \hat{\epsilon}(x_t, t, y) = \epsilon_\theta(x_t, t, \phi) + \gamma \left( \epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \phi) \right) $ or in the paper: $
\hat{\epsilon}(x_t, t, y) = (1 - \gamma)\,\epsilon_\theta(x_t, t, \phi) + \gamma\,\epsilon_\theta(x_t, t, y) $.

This achieves the same effect as classifier guidance, but without requiring a separate classifier.

The main purpose is to improve quality of conditional diffusion models without needing a separate, pre-trained classifier.

The fundamental problem with diffusion (unconditional generation) is that diffusion goes from noise to image step by step. This means it is highly unsuitable to generate a desired output because the generation cannot be controlled. So the question becomes: can you perturb the denoising trajectory?

Answer is classifier guidance generation. First, this means to train a noisy image classifier to generate a class label. Classifier output is $\nabla_{x_t} \log p(y \mid x_t)$. That is a noise component that is more aligned for a specific class.

What this yields is $\gamma \, \nabla_{x_t} \log p(y \mid x_t)$ where $\gamma > 1$ to amplify the signal. This shifts probability mass from the least likely to the most likely class values, meaning the greater the $\gamma$, the more class consistent the generation. $\gamma$ is the inverse temperature parameter.

So the old $\epsilon_\theta(x_t, t)$ now is:

$\hat{\epsilon}(x_t, t, y) = \epsilon_\theta(x_t, t) - \gamma \, \nabla_{x_t} \log p(y \mid x_t)$

More info: [paper](https://arxiv.org/pdf/2207.12598).

We basically sample gradients from a classifier when classifying an image of a desired class and feed that gradient to the diffusion model to ultimately perturb the model.

How to get class guidance without an independent classifier? Use the diffusion model itself to get perturbations.

Instead of training a separate classifier, we train a single diffusion model that can operate both conditionally and unconditionally. This is done using conditioning dropout, where the conditioning variable $y$ is randomly removed during training.

At inference time, we calculate both $\epsilon_\theta(x_t, t, y)$ and $\epsilon_\theta(x_t, t, \phi)$ using the same network. The difference between these implicitly captures the direction toward the class.

This leads to classifier-free guidance:

$\hat{\epsilon}(x_t, t, y) = \epsilon_\theta(x_t, t, \phi) + \gamma \left( \epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \phi) \right)$

or equivalently:

$\hat{\epsilon}(x_t, t, y) = (1 - \gamma)\,\epsilon_\theta(x_t, t, \phi) + \gamma\,\epsilon_\theta(x_t, t, y)$

This achieves the same effect as classifier guidance, but without requiring a separate classifier. The model trains itself.