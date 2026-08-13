"""E0: chord-threshold calibration (Annex B.1's up-front falsifier).

Direct evaluation of theory.chord_geometry across an (a, b) grid -- no
agents, no gossip, no stochastic simulation, no seed at all (chord_geometry
is a pure root-finding computation on the loss landscape's critical
points -- the "direct bisection on the AB chord" Annex B.1 specifies for
E0). Checks the complementarity identity (2.7a),

    vartheta_{A->B}(lambda) + vartheta_{B->A}(lambda) == 1,

which Lemma 2.4's clique-round projection (eq. 2.9) -- and therefore
B.0.1's "round boundaries are a sufficient statistic" claim that every
later H-phase experiment's logging economy depends on -- requires. This is
Annex B.1's E0 and B.6's "cheap up-front falsifier ... on which most of
Sec. 3 depends," meant to be checked before any other H-phase experiment
(starting with E7/E14, which consume the measured vartheta_dagger(lambda)
this module produces) is even queued -- see check_phaseh1.py /
gateh1_review.json.
"""
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from . import theory


@dataclass
class ChordCalibrationConfig:
    """One (a, b, r) grid point to evaluate theory.chord_geometry at."""

    name: str
    a: float
    b: float
    r: float = 1.0
    delta_norm: float = 1.0


@dataclass
class ChordCalibrationRun:
    """theory.chord_geometry's output at one grid point, plus the
    complementarity check (2.7a)."""

    name: str
    a: float
    b: float
    r: float
    lambda_: float
    vartheta_A_to_B: float
    vartheta_B_to_A: float
    complementarity_error: float  # |vartheta_A_to_B + vartheta_B_to_A - 1|

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ChordCalibrationRun":
        with open(path, "rb") as f:
            return pickle.load(f)


def run_chord_calibration(config: ChordCalibrationConfig) -> ChordCalibrationRun:
    geom = theory.chord_geometry(config.a, config.b, config.delta_norm, r=config.r)
    vab, vba = geom["vartheta_A_to_B"], geom["vartheta_B_to_A"]
    return ChordCalibrationRun(
        name=config.name, a=config.a, b=config.b, r=config.r,
        lambda_=geom["lambda"], vartheta_A_to_B=vab, vartheta_B_to_A=vba,
        complementarity_error=abs(vab + vba - 1.0),
    )


def experiment_E0(
    a: float = 0.5,
    # Spans lambda from 0 up to ~0.98 at a=0.5 (lambda -> 1 as
    # b -> a*theory.KAPPA_PHI ~= 0.0481 at this a; deliberately stays just
    # under the saddle-node limit rather than at it, since basin A ceases
    # to exist at lambda=1 and chord_geometry's root-finding is undefined
    # there -- H-chord's own single-crossing hypothesis, Annex D.1).
    b_list: List[float] = (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.047),
    r: float = 1.0,
) -> List[ChordCalibrationConfig]:
    """E0: bisect the AB chord across a lambda grid at fixed a, reading
    vartheta_{A->B} and vartheta_{B->A} independently -- Annex B.1's
    complementarity falsifier.
    """
    return [
        ChordCalibrationConfig(name=f"E0/a={a}/b={b}", a=a, b=b, r=r)
        for b in b_list
    ]
