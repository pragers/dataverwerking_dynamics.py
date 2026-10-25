import csv
import os
from typing import cast

import numpy as np
import polars as pl
import xy

# Measurments with file nr of first measurement and amount of measurements.
MEASUREMENTS = {1:(4670,6),2:(4680,6),3:(4688,6),4:(4697,5),5:(4708,5)}
MEASUREMENT_NRS = [4670,4671,4672,4673,4675,4676,4680,4681,4682,4683,4685,4687,4688,4690,4692,4694,4695,4697,4698,4699,4700,4701,4707,4708,4709,4711,4712,4714,4715,4716,4719]

## CONSTANTS
BALL_WEIGHT = 150e-3 # in kg
BALL_WEIGHT_ERROR = 1e-3 # in kg
GRAVITATIONAL_ACCELERATION = 9.81 # in m/s
HOCKEY_STICK_INERTIA_MOMENT = 0.165 # in kgm^2
HOCKEY_STICK_INERTIA_MOMENT_ERROR = 0.00825  # 5%

RESOLUTION_ORIGINAL = 480 #Width resolution of raw data
RESOLUTION_RESCALED = 1920 #Width resolution of ball data
CM_SCALING_FACTOR = 1.7014 #in pixels/cm
PIXEL_SCALING_FACTOR = RESOLUTION_ORIGINAL / (RESOLUTION_RESCALED * CM_SCALING_FACTOR) # used to convert from ball tracking data to cm
TRACKING_ACCURACY = 0.5 # in pixels
TRACKING_ACCURACY_METERS = 0.5/(CM_SCALING_FACTOR*100) # in meters



## Amount of frames over which we calculate the difference in position.
DIFF_N = 5
ROLLING_MEAN_N = 5
FPS = 1200

RADIUS = 0.95 # in m
RADIUS_ERROR = 0.5e-3 # in m

def main() -> None:

    process_q_values(os.getcwd())

def convert_raw_data(project_dir):
    """
        Convert raw data to csv files with q, k_before, k_after, u_height and k_stick fields

        Parameters:
            project_dir: location of the project files.
    """
    ## We have 5 sticks, loop through them and get the q for all measurements. save them to csv file for later use.
    for metingnr in range(1,6):
        start_measurement, amount_measurements = MEASUREMENTS[metingnr]
        start_measurement_index = MEASUREMENT_NRS.index(start_measurement)
        with open(f"{project_dir}\\output\\output_meting_{metingnr}.csv", 'w+') as f:
            writer = csv.writer(f, lineterminator="\n")
            header = ["q","k_before","k_after","u_height","k_stick","q_error", "k_before_error", "k_after_error", "u_height_error", "k_stick_error"]
            writer.writerow(header)
            for i in range(amount_measurements):
                q, k_before, k_after, u_height, k_stick, q_error, k_before_error, k_after_error, u_height_error, k_stick_error = get_q(project_dir, metingnr, start_measurement_index + i, i+1)
                print(f""" Measurement: {MEASUREMENT_NRS[start_measurement_index + i]}\n
                        Ratio of energy lost: {q}\n
                        Kinetic energy before collision: {k_before} joule\n
                        Kinetic energy after collision: {k_after} joule\n
                        Hockey stick rotation energy: {k_stick} joule\n
                        Ball height energy energy: {u_height} joule\n
                    """)
                writer.writerow([q, k_before, k_after, u_height, k_stick, q_error, k_before_error, k_after_error, u_height_error, k_stick_error])



def get_q(project_dir, stick_nr : int, measurement_index : int, measurement_nr: int):
    """
        Calculates the kinetic energy before, kinetic energy after, height energy and rotation energy after.
        The ratio between them gives q.

        Parameters:
            stick_nr: int, the number of the current stick (between 1 and 5)
            measurement_index: index of the current measurement in the MEASUREMENT_NRS.
            measurement_nr: number to complete the file name "meting x.csv"

        Returns:
            project_dir: location of the project files.
            q: ratio of energy lost
            k_before: kinetic energy before collision
            k_after: kinetic energy of the ball after collision
            u_height: potential energy from height after collision
            k_stick: kinetic energy from rotation of the hockey stick.
    """
    current_measurement = MEASUREMENT_NRS[measurement_index]
    ## Load in the data from csv.
    ball_data_before = pl.read_csv(f"{project_dir}\\ball_data_before\\data_DSC_{current_measurement}_before.csv", separator = ",")
    ball_data_after = pl.read_csv(f"{project_dir}\\ball_data_after\\data_DSC_{current_measurement}_after.csv", separator = ",")
    hockey_data = pl.read_csv(f"{project_dir}\\hockey_stick_data\\stick{stick_nr}\\meting {measurement_nr}.csv", separator = ",")
    ball_data_before = ball_data_before.unique(maintain_order=True)
    ball_data_after = ball_data_after.unique(maintain_order=True)
    ## Data from ball tracking has a different scale to the data from hockey stick tracking.
    ## we convert between them by converting back to the original pixels and applying the scaling measured in Fuij.
    ## Also immediately convert from cm to m by dividing by 100.
    ball_data_before *= PIXEL_SCALING_FACTOR / 100
    ball_data_after *= PIXEL_SCALING_FACTOR / 100
    hockey_data = hockey_data.with_columns([(pl.col(["X","Y"]) / 100)])

    ## Get the mean velocity before collision, since we said air resistance is negligable
    ## there are no forces acting (horizontally) on our ball so the speed should be constant.
    ## Divide by DIFF_N and multiply by FPS to get per seconds instead of per N frames.
    velocities = ball_data_before["x"].diff(n = DIFF_N) / DIFF_N * FPS
    velocities_error = velocities.rolling_std(window_size= ROLLING_MEAN_N).mean()
    rolling_mean_velocities =  velocities.rolling_mean(window_size = ROLLING_MEAN_N).drop_nulls()
    mean_velocity_before = cast(float,rolling_mean_velocities[-1])
    assert isinstance(mean_velocity_before, (float))
    assert isinstance(velocities_error, (float))
    energy_before = 0.5 * BALL_WEIGHT * mean_velocity_before ** 2
    energy_before_error = np.sqrt((BALL_WEIGHT_ERROR/BALL_WEIGHT)**2 + (velocities_error/mean_velocity_before))

    ## Get the kinetic energy of the ball.
    ball_x_velocity = ball_data_after["x"].diff(n = DIFF_N) / DIFF_N * FPS
    ball_y_velocity = ball_data_after["y"].diff(n = DIFF_N) / DIFF_N * FPS
    ball_velocity_squared_array =  ball_x_velocity ** 2 + ball_y_velocity ** 2
    ball_velocity_array = ball_velocity_squared_array.sqrt()
    ball_velocity = np.sqrt(ball_velocity_squared_array.rolling_mean(window_size = ROLLING_MEAN_N).drop_nulls()[0])
    ball_kinetic_energy = 0.5 * BALL_WEIGHT * ball_velocity**2
    assert isinstance(ball_kinetic_energy, (float))

    ball_velocity_error = ball_velocity_array.rolling_std(window_size= ROLLING_MEAN_N).mean()

    ## Calculate the energy going to height.
    ## Videos are inverted and y goes down so we take the difference the same way as normal.
    rolling_mean_height_after = ball_data_after["y"].rolling_mean(window_size = ROLLING_MEAN_N).drop_nulls()
    rolling_mean_height_before = ball_data_before["y"].rolling_mean(window_size = ROLLING_MEAN_N).drop_nulls()
    height_difference =  cast(float, rolling_mean_height_after[0]) - cast(float, rolling_mean_height_before[-1])
    height_energy_gained = BALL_WEIGHT * GRAVITATIONAL_ACCELERATION * height_difference
    height_energy_gained_error = height_energy_gained * np.sqrt((ball_velocity_error/ball_velocity)**2 + (BALL_WEIGHT_ERROR/BALL_WEIGHT)**2)


    assert isinstance(ball_velocity_error, (float))
    ball_kinetic_energy_error = ball_kinetic_energy* np.sqrt((2*ball_velocity_error/ball_velocity)**2 + (BALL_WEIGHT_ERROR/BALL_WEIGHT)**2)

    ## Calculate the total velocity of the hockey stick, by taking the pythagorean sum of the components.
    hockey_stick_x_velocity = hockey_data["X"].diff(n = DIFF_N) / DIFF_N * FPS
    hockey_stick_y_velocity = hockey_data["Y"].diff(n = DIFF_N) / DIFF_N * FPS
    hockey_stick_velocity = (hockey_stick_x_velocity ** 2 + hockey_stick_y_velocity ** 2).sqrt()
    mean_hockey_stick_velocity = cast(float, hockey_stick_velocity.rolling_mean(window_size = ROLLING_MEAN_N).drop_nulls()[0])

    hockey_stick_velocity_error = ball_velocity_squared_array.rolling_std(window_size= ROLLING_MEAN_N).mean()
    assert isinstance(hockey_stick_velocity_error, (float))
    ## Calculate rotation energy for the hockey stick.
    angular_velocity = mean_hockey_stick_velocity/RADIUS
    angular_velocity_relative_error = np.sqrt((hockey_stick_velocity_error/mean_hockey_stick_velocity)**2 + (RADIUS_ERROR/RADIUS)**2)
    assert isinstance(hockey_stick_velocity_error, (float))
    inertia_moment_stick, inertia_moment_relative_error = get_hockey_stick_inertia_moment(stick_nr)

    hockey_stick_rotation_energy = 0.5 * inertia_moment_stick * angular_velocity**2
    hockey_stick_rotation_energy_error = hockey_stick_rotation_energy * np.sqrt((2*angular_velocity_relative_error)**2 + inertia_moment_relative_error**2)

    total_energy_after = ball_kinetic_energy+hockey_stick_rotation_energy + height_energy_gained
    total_energy_after_error = hockey_stick_rotation_energy_error + ball_kinetic_energy_error + height_energy_gained_error

    q_error = total_energy_after / energy_before * np.sqrt((total_energy_after_error/total_energy_after)**2 + (energy_before_error/energy_before)**2)
    return (
        1 - total_energy_after / energy_before,
        energy_before,
        ball_kinetic_energy,
        height_energy_gained,
        hockey_stick_rotation_energy,
        q_error,
        energy_before_error,
        ball_kinetic_energy_error,
        height_energy_gained_error,
        hockey_stick_rotation_energy_error,
    )


def get_hockey_stick_inertia_moment(stick_nr : int):
    return HOCKEY_STICK_INERTIA_MOMENT, HOCKEY_STICK_INERTIA_MOMENT_ERROR



def measurement_err(project_dir) :
    ## Get the computer vision data and human data for 4707
    cv_data =  pl.read_csv(f"{project_dir}\\ball_data_before\\data_DSC_4707_before.csv", separator = ",")
    human_data =  pl.read_csv(f"{project_dir}\\Results2.csv", separator = ",")

    # Remove duplicate last frame and rescale.
    cv_data = cv_data.unique(maintain_order=True)
    cv_data *= PIXEL_SCALING_FACTOR

    # Get coordinates for edge of the ball by adding half the width to the center.
    cv_plus_width = cv_data["x"] + cv_data["width"]/2

    # Get the index for which the edge coordinate matches the first human coordinate by getting the index with the least square error.
    index = ((cv_plus_width - human_data["X"][0])**2).arg_min()
    assert isinstance(index, int)

    print(index)
    ## get the speed values for the computer and human data, the speed should stay roughly equal since there are no forces horizontally.
    cv_values = cv_data["x"].diff()[index:]
    human_speed = human_data["X"].diff()
    print(cv_values, human_speed[:cv_values.len()])
    ## take the difference between the speed values, squared
    mean_square_error = (cv_values - human_speed[:cv_values.len()])**2
    print(mean_square_error.mean())
    # get the rolling standard deviation to check the internal validity of the measurements.
    # Since there are basically no forces acting the standard deviation should reveal the uncertainty in the measurement.
    # Using a rolling std we dont compare the start speed to end speed, this minimizes the effect of any actual forces acting on the ball.
    print(human_speed.rolling_std(window_size = 15).mean(), cv_data["x"].diff().rolling_std(window_size = 15).mean())

def get_maximum_comp_vision_std(project_dir,window_size):
    max_seen = 0
    seen_on= 0
    for metingnr in range(1,6):
        start_measurement, amount_measurements = MEASUREMENTS[metingnr]
        start_measurement_index = MEASUREMENT_NRS.index(start_measurement)
        for i in range(amount_measurements):
            measurement_nr =MEASUREMENT_NRS[start_measurement_index + i]
            ball_data_before = pl.read_csv(f"{project_dir}\\ball_data_before\\data_DSC_{measurement_nr}_before.csv", separator = ",")
            ball_data_before = ball_data_before.unique(maintain_order=True)
            ball_data_before *= PIXEL_SCALING_FACTOR

            mean_std = ball_data_before["x"].diff().rolling_std(window_size=window_size).mean()
            print(measurement_nr, mean_std)
            if isinstance(mean_std,float) and mean_std> max_seen :
                max_seen = mean_std
                seen_on = measurement_nr
    return max_seen, seen_on


def process_q_values(project_dir):
    results = pl.DataFrame(
        schema={
            "q_mean": pl.Float64,
            "q_error": pl.Float64,
        }
    )
    for i in range(1,6):
        measurement_data = pl.read_csv(f"{project_dir}\\output\\output_meting_{i}.csv", separator = ",")
        q = measurement_data["q"]
        q_error = measurement_data["q_error"]
        weights = 1/ q_error**2
        q_mean = (q*weights).sum() / weights.sum()  # pyright: ignore[reportOperatorIssue]
        q_mean_error = (1/weights.sum())**0.5   # pyright: ignore[reportOperatorIssue]
        data_frame = pl.DataFrame({
            "q_mean": q_mean,
            "q_error": q_mean_error,
        })
        results.extend(data_frame)
    results.write_csv(f"{project_dir}\\output\\total_output.csv")
    indices = [i for i in range(1,6)]
    chart = xy.errorbar_chart(
        xy.bar(
            indices,
            results["q_mean"].to_numpy(),
            color="#547BA5",
            opacity=0.9,
            corner_radius=4,
            width=0.65,
        ),
        xy.errorbar(
            indices,
            results["q_mean"].to_numpy(),
            yerr=results["q_error"].to_numpy(),
            color="#202020",
            width=2,
        ),
        xy.x_axis(label="Hockey stick"),
        xy.y_axis(
            label="Mean energy loss fraction q",
            bounds=(0, 1),
        ),
        xy.theme(
            background="#FFFFFF",
            plot_background="#FFFFFF",
            grid_color="#E5E7EB",
            axis_color="#9CA3AF",
            text_color="#1F2937",
        ),
        title="Energy loss by hockey stick",
    )

    chart.to_png("output.png")
