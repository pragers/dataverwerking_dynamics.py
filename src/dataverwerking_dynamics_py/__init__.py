from typing import cast

import polars as pl


## CONSTANTS
BALL_WEIGHT = 100e-3 # in kg
GRAVITATIONAL_ACCELERATION = 9.81 # in m/s

RESOLUTION_ORIGINAL = 480 #Height resolution
RESOLUTION_RESCALED = 1920 #Width resolution
CM_SCALING_FACTOR = 1.7014 #in pixels/cm
PIXEL_SCALING_FACTOR = RESOLUTION_ORIGINAL / (RESOLUTION_RESCALED*CM_SCALING_FACTOR)



def main() -> None:
    ## Load in the data from csv.
    ball_data_before = pl.read_csv("", separator = ",")
    ball_data_after = pl.read_csv("", separator = ",")
    hockey_data = pl.read_csv("", separator = ",")

    ## Data from ball tracking has a different scale to the data from hockey stick tracking.
    ## we convert between them by converting back to the original pixels and applying the scaling measured in Fuij.
    ball_data_before *= PIXEL_SCALING_FACTOR
    ball_data_after *= PIXEL_SCALING_FACTOR

    ## get the mean velocity before collision, since we said airresistance is negligable 
    ## there are no forces acting (horizontally) on our ball so the speed should be constant.
    mean_velocity_before = cast(float, ball_data_before["x"].diff(null_behavior="ignore").mean())
    assert isinstance(mean_velocity_before, (float))
    energy_before = 0.5 * BALL_WEIGHT * mean_velocity_before**2

    ## Videos have height inverted, therefore we subtract the other way around.
    mean_height_difference = cast(float, ball_data_before["y"].mean()) - cast(float, ball_data_after["y"].mean())
    mean_ball_height_energy_gained = BALL_WEIGHT * GRAVITATIONAL_ACCELERATION * mean_height_difference

    ## Get the mean kinetic energy of the ball.
    kinetic_energy = 0.5 * BALL_WEIGHT * ball_data_after["x"].diff() ** 2
    mean_kinetic_energy = cast(float, kinetic_energy.mean())
    assert isinstance(mean_kinetic_energy, (float))
    
    