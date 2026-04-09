# cVAE Notes (from notebook prototype)

This README captures the theory notes originally written in the notebook, and is intended to live next to the model/training implementation.

## Notation

- `p_theta`: generative model parameterized by `theta`
- `q_phi`: variational approximation (encoder) parameterized by `phi`

## Variables

- Emissivity `e in R^41`: training target
- Latent variable `z in R^d`: unobserved
- Bolometry `b in R^8`: observed conditioning input

Given `b`, sample `z`, then generate `e`.

## Decoder

1. Sample latent `z ~ N(0, I)`.
2. Model emissivity with decoder-conditioned distribution `p_theta(e | b, z)`.
3. Use Gaussian likelihood with decoder mean and identity covariance:
   `p_theta(e | b, z) = N(mu_theta(b, z), I)`.

## Encoder

Approximate posterior:

`q_phi(z | e, b) = N(mu_z(e, b), diag(sigma_z^2(e, b)))`

The encoder outputs mean and variance for latent variables that could explain observed emissivity under conditioning `b`.

## Training

1. Encode `(e, b)` to `(mu_z, sigma_z)`.
2. Reparameterize:
   `z = mu_z + sigma_z * epsilon`, `epsilon ~ N(0, I)`.
3. Decode `(b, z)` to emissivity prediction `e_hat = mu_theta(b, z)`.

## Loss

- Reconstruction term: `||e - e_hat||^2`
- KL term: `KL(q_phi(z | e, b) || N(0, I))`
- Combined objective:
  `L = E_{q_phi(z|e,b)}[||e - e_hat||^2] + KL(q_phi(z|e,b) || N(0, I))`

## Inference

- Encoder is not used.
- Sample `z` from prior and decode conditioned on measurement `b`.
