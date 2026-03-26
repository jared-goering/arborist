"""Mutators — generate child configs from parent nodes."""

from arborist.mutators.llm_mutator import LLMMutator, _perturb_config_fallback


class RandomMutator:
    """Simple random perturbation mutator (wraps _perturb_config_fallback)."""

    def __init__(self, n_children: int = 2):
        self.n_children = n_children

    def __call__(self, config, results, context) -> list[dict]:
        return _perturb_config_fallback(config, n=self.n_children)


__all__ = ["LLMMutator", "RandomMutator"]
