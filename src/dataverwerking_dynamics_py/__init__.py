import polars as pl


BALL_WEIGHT = 100e-3 # in kg

RESOLUTION_ORIGINAL = 480 #Height resolution
RESOLUTION_RESCALED = 1920 #Width resolution
CM_SCALING_FACTOR = 1.7014 #in pixels/cm
PIXEL_SCALING_FACTOR = RESOLUTION_ORIGINAL / (RESOLUTION_RESCALED*CM_SCALING_FACTOR)

def main() -> None:
    ball_data_before = pl.read_csv("", separator = ",")
    ball_data_after = pl.read_csv("", separator = ",")
    hockey_data = pl.read_csv("", separator = ",")

    ball_data_before *= PIXEL_SCALING_FACTOR
    ball_data_after *= PIXEL_SCALING_FACTOR

    ball_diameter = ball_data_after["width"][0]
    
