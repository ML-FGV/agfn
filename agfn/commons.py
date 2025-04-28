from enum import Enum

class EdgeType(Enum):
    NO_RELATIONSHIP = 0
    LEFT_CAUSES_RIGHT = 1
    RIGHT_CAUSES_LEFT = 2
    LATENT_CONFOUNDING = 3

