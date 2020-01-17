from enum import Enum


class AdaptationLevel(Enum):
    BASELINE_A = 1
    BASELINE_B = 2
    BASELINE_C = 3
    BASELINE_D = 4



class Status(Enum):
    LEARNING_STARTED = 1
    LEARNING_DONE = 2
    ADAPT_STARTED = 3
    ADAPT_DONE = 4
    CHARGING_STARTED = 5
    CHARGING_DONE = 6
    AT_WAYPOINT = 7
    ONLINE_LEARNING_STARTED = 8
    ONLINE_LEARNING_DONE = 9
