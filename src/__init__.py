
"""
PVF-U: Unified Primitive Valence Foundation

A three-tiered architecture for affective AI that combines
solitary survival with equitable social negotiation.
"""

__version__ = "1.0.0"
__author__ = "Precious Adeniyi"

from pvf_u.core.agent import PVFAgent
from pvf_u.core.layers.layer0_interoceptive import InteroceptiveCore
from pvf_u.core.layers.layer1_relational_gate import RelationalGate
from pvf_u.core.layers.layer2_ecosystemic import EcosystemicSuperstructure

__all__ = [
    "PVFAgent",
    "InteroceptiveCore",
    "RelationalGate",
    "EcosystemicSuperstructure",
]
