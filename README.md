# AdversarialTracks
Here's some papers of adversarial attack in different kinds of tracks

## <a href='https://github.com/Anson-He/AdversarialTracks/blob/main/Disentangle/Deeply_Supervised_Discriminative_Learning_for_Adversarial_Defense.pdf'>Deeply Supervised Discriminative Learning for Adversarial Defense</a>
This paper shows a method to enhance the robustness of models by mapping the penultimate layer features into a lower feature space, then minimize the inner-class distance and maximize the inter-class distance throght Lpc restraint with help of a trainable class centroids.

## <a href='https://github.com/Anson-He/AdversarialTracks/blob/main/Disentangle/Adversarial%20Robustness%20through%20Disentangled%20Representations.pdf'>Adversarial Robustness through Disentangled Representations</a>
In this paper, the authors discovered that the class-irrelevant representation is the primary cause of models misclassifying adversarial examples. Therefore, they introduced a disentanglement framework to separate the class-specific and class-irrelevant representations, implying both robust and non-robust representations. They employed various loss functions to train this framework, enhancing its ability to defend against adversarial attacks.

## <a href='https://github.com/Anson-He/AdversarialTracks/blob/main/DDRM/Denoising%20Diffusion%20Restoration%20Models.pdf'>Denoising Diffusion Restoration Models</a>
This paper provide a more efficient way to calculate SVD of Toeplitz matrix of kernel.