The main purpose is to improve quality of conditional diffusion models without needing a separate, pre-trained classifier.

The fundamental problem with diffusion (unconditional generation) is that diffusion goes from noise to image step by step. This means it is highly unsuitable to generate a desired output because the generation cannot be controlled. So the question becomes: can you perturb the denoising trajectory?

Answer is classifier-free guidance generation. First, this means to train a noisy image classifier to generate a class label. Classifier output is $ \nabla_{x_t} \log p(y \mid x_t) $. That is a noise component that is more aligned for a specific class.

What this yields is $ \gamma \, \nabla_{x_t} \log p(y \mid x_t) $ where $ \gamma > 1  $ to amplify the signal. This shifts probability mass from the least likely to the most likely class values, meaning the greater the $ \gamma $, the more class-consistent the generation. $ \gamma $ is the inverse temperature parameter. For non-technical people, low $ \gamma $ would mean exploring new options and high $ \gamma $ would mean exploitation of the best option.

So the old $ \epsilon_\theta(x_t, t) $ now is $ \hat{\epsilon}(x_t, t, y) = \epsilon_\theta(x_t, t) - \gamma \, \nabla_{x_t} \log p(y \mid x_t) $; this an overtly high-level explanation of how this approximation is made. More info: https://arxiv.org/pdf/2207.12598.

We basically sample gradients from a classifier when classifying an image of a desired class and feed that gradient to the diffusion model to ultimately perturb the model.