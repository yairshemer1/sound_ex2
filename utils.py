import datetime

from enum import Enum


class Genre(Enum):
    """
    This enum class is optional and defined for your convinience, you are not required to use it.
    Please use the int labels this enum defines for the corresponding genras in your predictions.
    """

    CLASSICAL: int = 0
    HEAVY_ROCK: int = 1
    REGGAE: int = 2


def get_tme_now():
    now = datetime.datetime.now()
    time_str = now.strftime("%Y%m%d_%H%M")
    return time_str
