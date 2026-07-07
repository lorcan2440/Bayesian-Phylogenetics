# Bayesian Phylogenetics (Educational, Minimal)

This project is a first-principles, educational implementation of Bayesian phylogenetic reconstruction using:

- JC69 substitution model
- Poisson-process interpretation of substitutions along branches
- Metropolis-Hastings MCMC over `{T, t_T, mu}`

where:

- `T` is a rooted binary tree topology
- `t_T` are branch lengths
- `mu` is a global mutation rate

The app estimates empirical topology posterior probabilities `P(T | D)` from MCMC samples.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Then open `http://127.0.0.1:5000`.

## Input format

Provide aligned DNA sequences of equal length:

- FASTA, or
- one record per line: `name: ACGT...`

Allowed symbols are `A C G T`.

