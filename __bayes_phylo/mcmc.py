from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random

from .likelihood import build_leaf_sequences, log_likelihood
from .parsing import SequenceData
from .prior import log_prior
from .tree import MIN_BRANCH_LENGTH, PhyloTree, random_binary_tree


@dataclass(frozen=True)
class SamplerConfig:
    iterations: int = 4000
    burn_in: int = 1000
    thinning: int = 10
    seed: int | None = None

    init_branch_mean: float = 0.1
    branch_prior_rate: float = 10.0
    mu_prior_rate: float = 1.0

    topology_move_prob: float = 0.3
    branch_move_prob: float = 0.5
    branch_log_step: float = 0.35
    mu_log_step: float = 0.25


@dataclass(frozen=True)
class PosteriorTopology:
    topology: str
    probability: float
    count: int
    representative_newick: str


@dataclass(frozen=True)
class MCMCResult:
    sampled_states: int
    accepted_moves: dict[str, int]
    proposed_moves: dict[str, int]
    acceptance_rates: dict[str, float]
    topologies: list[PosteriorTopology]
    mu_posterior_mean: float
    map_newick: str
    map_log_posterior: float


def run_mcmc(data: SequenceData, config: SamplerConfig) -> MCMCResult:
    _validate_config(config)

    rng = random.Random(config.seed)
    data_by_taxon = {
        name: tuple(site_states)
        for name, site_states in zip(data.taxa, data.encoded)
    }

    current_tree, current_mu, current_log_like, current_log_prior = _init_valid_state(
        data=data,
        data_by_taxon=data_by_taxon,
        rng=rng,
        config=config,
    )
    current_log_post = current_log_like + current_log_prior

    map_tree = current_tree.copy()
    map_mu = current_mu
    map_log_post = current_log_post

    proposed_moves = {"topology": 0, "branch": 0, "mu": 0}
    accepted_moves = {"topology": 0, "branch": 0, "mu": 0}

    posterior_counts: Counter[str] = Counter()
    best_newick_for_topology: dict[str, tuple[float, str]] = {}
    sampled_states = 0
    mu_sum = 0.0

    for step in range(1, config.iterations + 1):
        move = _draw_move_type(rng, config)
        proposed_moves[move] += 1

        proposed_tree = current_tree
        proposed_mu = current_mu
        log_hastings = 0.0
        proposal_valid = True

        if move == "topology":
            proposed_tree = current_tree.copy()
            proposal_valid = proposed_tree.propose_nni(rng)
        elif move == "branch":
            proposed_tree = current_tree.copy()
            child = rng.choice(list(proposed_tree.branch_length.keys()))
            scale = math.exp(rng.uniform(-config.branch_log_step, config.branch_log_step))
            new_length = proposed_tree.branch_length[child] * scale
            if new_length <= MIN_BRANCH_LENGTH:
                proposal_valid = False
            else:
                proposed_tree.branch_length[child] = new_length
                log_hastings = math.log(scale)
        else:  # "mu"
            scale = math.exp(rng.uniform(-config.mu_log_step, config.mu_log_step))
            proposed_mu = current_mu * scale
            if proposed_mu <= 0.0:
                proposal_valid = False
            else:
                log_hastings = math.log(scale)

        if proposal_valid:
            proposed_leaf_sequences = build_leaf_sequences(proposed_tree, data_by_taxon)
            proposed_log_like = log_likelihood(
                proposed_tree, proposed_leaf_sequences, data.n_sites, proposed_mu
            )
            proposed_log_prior = log_prior(
                proposed_tree,
                proposed_mu,
                branch_rate=config.branch_prior_rate,
                mu_rate=config.mu_prior_rate,
            )
            proposed_log_post = proposed_log_like + proposed_log_prior

            log_alpha = proposed_log_post - current_log_post + log_hastings
            u = rng.random()
            if log_alpha >= 0.0 or u < math.exp(log_alpha):
                current_tree = proposed_tree
                current_mu = proposed_mu
                current_log_like = proposed_log_like
                current_log_prior = proposed_log_prior
                current_log_post = proposed_log_post
                accepted_moves[move] += 1

                if current_log_post > map_log_post:
                    map_log_post = current_log_post
                    map_tree = current_tree.copy()
                    map_mu = current_mu

        if step > config.burn_in and (step - config.burn_in) % config.thinning == 0:
            sampled_states += 1
            mu_sum += current_mu
            topology = current_tree.topology_signature()
            posterior_counts[topology] += 1

            current_newick = current_tree.to_newick(include_lengths=True)
            best = best_newick_for_topology.get(topology)
            if best is None or current_log_post > best[0]:
                best_newick_for_topology[topology] = (current_log_post, current_newick)

    topologies = _summarize_topologies(
        counts=posterior_counts,
        representative=best_newick_for_topology,
        sampled_states=sampled_states,
    )
    acceptance_rates = {
        key: (
            accepted_moves[key] / proposed_moves[key] if proposed_moves[key] else 0.0
        )
        for key in proposed_moves
    }
    mu_mean = mu_sum / sampled_states if sampled_states else current_mu
    map_newick = map_tree.to_newick(include_lengths=True)

    return MCMCResult(
        sampled_states=sampled_states,
        accepted_moves=accepted_moves,
        proposed_moves=proposed_moves,
        acceptance_rates=acceptance_rates,
        topologies=topologies,
        mu_posterior_mean=mu_mean,
        map_newick=map_newick,
        map_log_posterior=map_log_post,
    )


def _init_valid_state(
    data: SequenceData,
    data_by_taxon: dict[str, tuple[int, ...]],
    rng: random.Random,
    config: SamplerConfig,
) -> tuple[PhyloTree, float, float, float]:
    for _ in range(200):
        tree = random_binary_tree(
            taxa=list(data.taxa),
            rng=rng,
            init_branch_mean=config.init_branch_mean,
        )
        mu = max(MIN_BRANCH_LENGTH, rng.expovariate(config.mu_prior_rate))
        leaf_sequences = build_leaf_sequences(tree, data_by_taxon)
        log_like = log_likelihood(tree, leaf_sequences, data.n_sites, mu)
        log_p = log_prior(
            tree,
            mu,
            branch_rate=config.branch_prior_rate,
            mu_rate=config.mu_prior_rate,
        )
        if math.isfinite(log_like) and math.isfinite(log_p):
            return tree, mu, log_like, log_p
    raise RuntimeError("Could not initialize a valid MCMC state.")


def _draw_move_type(rng: random.Random, config: SamplerConfig) -> str:
    u = rng.random()
    if u < config.topology_move_prob:
        return "topology"
    if u < config.topology_move_prob + config.branch_move_prob:
        return "branch"
    return "mu"


def _summarize_topologies(
    counts: Counter[str],
    representative: dict[str, tuple[float, str]],
    sampled_states: int,
) -> list[PosteriorTopology]:
    if sampled_states <= 0:
        return []

    rows: list[PosteriorTopology] = []
    for topology, count in counts.most_common():
        prob = count / sampled_states
        rep_newick = representative[topology][1]
        rows.append(
            PosteriorTopology(
                topology=topology,
                probability=prob,
                count=count,
                representative_newick=rep_newick,
            )
        )
    return rows


def _validate_config(config: SamplerConfig) -> None:
    if config.iterations <= 0:
        raise ValueError("iterations must be > 0")
    if config.burn_in < 0:
        raise ValueError("burn_in must be >= 0")
    if config.burn_in >= config.iterations:
        raise ValueError("burn_in must be smaller than iterations")
    if config.thinning <= 0:
        raise ValueError("thinning must be > 0")
    if config.init_branch_mean <= 0.0:
        raise ValueError("init_branch_mean must be > 0")
    if config.branch_prior_rate <= 0.0:
        raise ValueError("branch_prior_rate must be > 0")
    if config.mu_prior_rate <= 0.0:
        raise ValueError("mu_prior_rate must be > 0")
    if config.branch_log_step <= 0.0 or config.mu_log_step <= 0.0:
        raise ValueError("Proposal step sizes must be > 0.")
    if config.topology_move_prob < 0.0 or config.branch_move_prob < 0.0:
        raise ValueError("Move probabilities must be non-negative.")
    if config.topology_move_prob + config.branch_move_prob >= 1.0:
        raise ValueError(
            "topology_move_prob + branch_move_prob must be < 1.0 "
            "(remaining mass is for mu moves)."
        )
