"""Modelos multiestado de Markov para Vida y Salud."""

from __future__ import annotations

from .multistate import MultiStateModel, constant_generator, illness_death_generator

__all__ = ["MultiStateModel", "constant_generator", "illness_death_generator"]
