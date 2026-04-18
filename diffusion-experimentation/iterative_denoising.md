Diffusion models learn a reverse process starting from noise to image as a sequence of transformations. Why can't we do it one step?

All at once from pure noise is impossible to stabilize in one pass, and far too slow and computationally expensive. When the number of noise scales approaches infinity, we essentially perturb the distribution of data with continuously growing levels of noise. Then, the noise perturbation procedure is a **continuous-time stochastic process**. The math is very complex...

Really any generative image process is a stochastic process (high-level explanation): drift term + diffusion term. In diffusion, the drift term is what pushes generation towards noise during the forward process and towards structure in the reverse process. The diffusion term is some fuction $ g(t) $ for random noise with tiny Gaussian noise steps. This is better explained in Equation 6 & 7 in the paper: https://arxiv.org/pdf/2006.11239.

<img src="photos/image1.png" width="500" />

Solving a reverse SDE yields a score based generative model. Transforming data to a noise distribution can be done with an SDE. It can be reversed to generate samples from noise if we know the score of the distribution at each time step. Again, the math is very complex...
