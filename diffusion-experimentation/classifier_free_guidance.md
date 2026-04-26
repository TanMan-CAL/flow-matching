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

This achieves the same effect as classifier guidance, but without requiring a separate classifier.