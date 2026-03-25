"""Mutators — generate child configs from parent nodes."""

from arborist.tree import _perturb_config as _perturb

from arborist.mutators.llm_mutator import LLMMutator


class RandomMutator:
    """Simple random perturbation mutator (wraps _perturb_config)."""

    def __init__(self, n_children: int = 2):
        self.n_children = n_children

    def __call__(self, config, results, context) -> list[dict]:
        children = []
        while len(children) < self.n_children:
            children.extend(_perturb(config))
        return children[: self.n_children]


__all__ = ["LLMMutator", "RandomMutator"]
