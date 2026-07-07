## Bayesian Phylogenetics

This repo aims to implement Bayesian phylogenetic inference from first-principles in Python.

### Progress

- [x] Implement a phylogenetic tree data structure
- [x] Implement the GTR+Γ mutation model
- [x] Implement Felsenstein's pruning algorithm for likelihood calculation
- [ ] Implement Metropolis-Hastings algorithm
- [ ] Implement Markov chain Monte Carlo (MCMC) with burn-in and pruning at stationarity
- [ ] Calculate the posterior trees and the MAP tree
- [ ] Calculate clade consensus probabilities
- [ ] Convert to MC^3 using multiprocessing for parallel tempering
- [ ] Use thermodynamic integration to calculate the evidence P(D)
- [ ] Perform ancestral state reconstruction

### Mutation model

This code uses the GTR+Γ (general time reversible with gamma-distributed rate heterogeneity) model of nucleotide substitution. The GTR model is a continuous-time Markov chain (CTMC) model that allows for different mutation rates between each pair of nucleotides (A, C, G, T). The gamma distribution accounts for the fact that different sites in the sequence may evolve at different rates.

The GTR+Γ model is one of the most sophisticated models of mutation rates, more general than simpler methods like JC69 or K2P.

The GTR+Γ mutation model has 11 independent parameters:

- **Base frequencies**: $\pi_A, \pi_C, \pi_G, \pi_T$ (the equilibrium frequencies of each nucleotide)
- **Rate parameters**: $r_{AC}, r_{AG}, r_{AT}, r_{CG}, r_{CT}, r_{GT}$ (the relative rates of substitution between each pair of nucleotides)
- **Alpha**: $\alpha$ (the shape parameter of the gamma distribution for rate heterogeneity)

The transition rate matrix for GTR is:

$$ \mathbf{Q} = \begin{bmatrix} 
* & r_{AC} \pi_C & r_{AG} \pi_G & r_{AT} \pi_T \\ 
r_{AC} \pi_A & * & r_{CG} \pi_G & r_{CT} \pi_T \\ 
r_{AG} \pi_A & r_{CG} \pi_C & * & r_{GT} \pi_T \\ 
r_{AT} \pi_A & r_{CT} \pi_C & r_{GT} \pi_G & * \\ 
\end{bmatrix} $$

where $*$ denotes the diagonal elements, which are set such that each row sums to zero. The corresponding discrete-time transition probability matrix is given by the matrix exponential:

$$ \mathbf{P}(t) = e^{\mathbf{Q} \gamma_k t} $$

where $t$ is the branch length (time) and $\mathbf{P}(t)$ gives the probabilities of transitioning from one nucleotide to another over time $t$. The factor $\gamma_k$ is the rate modifier for nucleotide site $k$, drawn from a Gamma distribution:

$$ \gamma_k \sim \text{Gamma}(\alpha, \alpha) $$

The gamma distribution allows for modeling rate heterogeneity across sites, meaning that some sites may evolve faster or slower than others.

In the code, this matrix exponential is computed by diagonalising $\mathbf{Q}$ and using the eigenvalues and eigenvectors to compute $e^{\mathbf{Q} \gamma_k t}$ efficiently, re-using partial values to avoid redundant calculations.

### Felsenstein's pruning algorithm

To calculate the likelihood of observing the sequences at the extant nodes of the tree, we use Felsenstein's pruning algorithm, which involves dynamic programming. We use the fact that the likelihood of an ancestral node factorises as a product involving the likelihoods of its child nodes:

$$ L_{\text{ancestor}}^{(k)}(x) = \left( \sum_{y \in \{A, C, G, T\}} P_{xy}(t_1) L_{\text{descendant 1}}^{(k)}(y) \right) \times \left( \sum_{z \in \{A, C, G, T\}} P_{xz}(t_2) L_{\text{descendant 2}}^{(k)}(z) \right) $$

where $L_{\text{ancestor}}^{(k)}(x)$ is the likelihood of observing the sequences at all descendant nodes of the ancestor node at site $k$, given that the ancestor has nucleotide $x$. This DP algorithm is implemented by a post-order traversal of the tree, storing each likelihood vector for each node as we go.

The overall likelihood of the tree is then found based on the likelihoods at the root node:

$$ p(D_k | T, b, \theta, \gamma_k) = \sum_{x \in \{A, C, G, T\}} \pi_x L_{\text{root}}^{(k)}(x) $$

where $p(D_k | T, b, \theta, \gamma_k)$ is the likelihood of observing the data at site $k$, given the tree $T$, branch lengths $b$, GTR parameters $\theta$, and rate modifier $\gamma_k$.

We then marginalise over the gamma distribution to get the overall likelihood of the data at site $k$:

$$ p(D_k | T, b, \theta, \alpha) = \int_0^\infty p(D_k | T, b, \theta, \gamma_k) f(\gamma_k | \alpha) d\gamma_k $$

where $f(\gamma_k | \alpha)$ is the probability density function (PDF) of the gamma distribution with shape parameter $\alpha$. In practice, this integral is approximated using a discrete sum over a set of gamma values at equal-probability intervals of the gamma distribution.

Finally, we assume independence across sites to get the overall likelihood of the data:

$$ p(D | T, b, \theta, \alpha) = \prod_{k=1}^{} p(D_k | T, b, \theta, \alpha) $$

For notational simplicity we absorb the parameter $\alpha$ into the mutation model parameters $\theta$ and refer to this output as

$$ p(D | T, b, \theta). $$