import csv
import os
from typing import cast

import numpy as np
import polars as pl
import xy
from lmfit import Model



# Measurments with file nr of first measurement and amount of measurements.
MEASUREMENTS = {1:(4670,6),2:(4680,6),3:(4688,6),4:(4697,5),5:(4708,5)}
MEASUREMENT_NRS = [4670,4671,4672,4673,4675,4676,4680,4681,4682,4683,4685,4687,4688,4690,4692,4694,4695,4697,4698,4699,4700,4701,4707,4708,4709,4711,4712,4714,4715,4716,4719]

#Placeholder values
HOCKEY_STICK_INERTIA_MOMENT = 0.165 # in kgm^2
HOCKEY_STICK_INERTIA_MOMENT_ERROR = 0.00825  # 5%
HOCKEY_STICK_INERTIA_MOMENTS = [HOCKEY_STICK_INERTIA_MOMENT for _ in range(5)]
HOCKEY_STICK_INERTIA_MOMENT_ERRORS = [HOCKEY_STICK_INERTIA_MOMENT_ERROR for _ in range(5)]

## Constants
BALL_WEIGHT = 165e-3 # in kg
BALL_WEIGHT_ERROR = 5e-3 # in kg
GRAVITATIONAL_ACCELERATION = 9.81 # in m/s^2

## Video related constants
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

## Placeholders
RADIUS = 0.95 # in m
RADIUS_ERROR = 0.5e-3 # in m

def main() -> None:
    convert_raw_data(os.getcwd())
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

    ## Remove exact duplicates.
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
    def get_velocities(places, error):
        ## Get the time and weights for the model.
        time = np.array([i/FPS for i in range(places.len())])
        weights = np.array([1/error for _ in range(places.len())])

        ## Create the model, function is expected to be linear.
        velocity_model = Model(linear)
        params = velocity_model.make_params(a=0,b=1)

        ## Fit the data, the angular velocity should be the slope or "a" variable.
        result = velocity_model.fit(places.to_numpy(), params, x=time, weights = weights)
        slope_parameter = result.params.get("a")

        return slope_parameter.value, slope_parameter.stderr
    velocity, velocity_error = get_velocities(ball_data_before["x"], TRACKING_ACCURACY_METERS)
    # velocities = ball_data_before["x"].diff(n = DIFF_N) / DIFF_N * FPS
    # velocities_error = velocities.rolling_std(window_size= ROLLING_MEAN_N).mean()

    # rolling_mean_velocities =  velocities.rolling_mean(window_size = ROLLING_MEAN_N).drop_nulls()
    # mean_velocity_before = cast(float,rolling_mean_velocities[-1])
    # assert isinstance(mean_velocity_before, (float))
    # assert isinstance(velocities_error, (float))
    energy_before = 0.5 * BALL_WEIGHT * velocity ** 2
    energy_before_error = energy_before * np.sqrt((BALL_WEIGHT_ERROR/BALL_WEIGHT)**2 + (2*velocity_error/velocity)**2)

    ## Get the kinetic energy of the ball.
    #
    ball_place = (ball_data_after["x"]**2 + ball_data_after["y"]**2).sqrt()
    ball_error = np.sqrt(2) * TRACKING_ACCURACY_METERS
    ball_velocity, ball_velocity_error = get_velocities(ball_place, ball_error)

    # ##########################
    # ball_x_velocity = ball_data_after["x"].diff(n = DIFF_N) / DIFF_N * FPS
    # ball_y_velocity = ball_data_after["y"].diff(n = DIFF_N) / DIFF_N * FPS
    # ball_velocity_squared_array =  ball_x_velocity ** 2 + ball_y_velocity ** 2
    # ball_velocity_array = ball_velocity_squared_array.sqrt()

    # ball_velocity = np.sqrt(ball_velocity_squared_array.rolling_mean(window_size = ROLLING_MEAN_N).drop_nulls()[0])
    # ball_velocity_error = ball_velocity_array.rolling_std(window_size= ROLLING_MEAN_N).mean()
    # assert isinstance(ball_velocity_error, float)
    # ###########################

    ball_kinetic_energy = 0.5 * BALL_WEIGHT * ball_velocity**2
    energy_before_error = ball_kinetic_energy * np.sqrt((BALL_WEIGHT_ERROR/BALL_WEIGHT)**2 + (2*ball_velocity_error/ball_velocity)**2)
    assert isinstance(ball_kinetic_energy, (float))



    ## Calculate the energy going to height.
    ## Videos are inverted and y goes down so we take the difference the same way as normal.
    rolling_mean_height_after = ball_data_after["y"].rolling_mean(window_size = ROLLING_MEAN_N).drop_nulls()
    rolling_mean_height_before = ball_data_before["y"].rolling_mean(window_size = ROLLING_MEAN_N).drop_nulls()
    height_difference =  cast(float, rolling_mean_height_after[0]) - cast(float, rolling_mean_height_before[-1])
    height_energy_gained = BALL_WEIGHT * GRAVITATIONAL_ACCELERATION * height_difference
    height_energy_gained_error = np.sqrt(
        (GRAVITATIONAL_ACCELERATION
         * height_difference * BALL_WEIGHT_ERROR)**2
        + (BALL_WEIGHT
           * GRAVITATIONAL_ACCELERATION
           * ball_velocity_error)**2
    )


    assert isinstance(ball_velocity_error, (float))
    ball_kinetic_energy_error = ball_kinetic_energy* np.sqrt((2*ball_velocity_error/ball_velocity)**2 + (BALL_WEIGHT_ERROR/BALL_WEIGHT)**2)

    ## Calculate the total velocity of the hockey stick, by taking the pythagorean sum of the components.
    hockey_stick_place = (hockey_data["X"] ** 2 + hockey_data["Y"] ** 2).sqrt()
    hockey_stick_place_error = np.sqrt(2) * TRACKING_ACCURACY_METERS
    hockey_stick_angle = hockey_stick_place/RADIUS
    hockey_stick_angle_error = hockey_stick_place_error/RADIUS

    def get_angular_velocity():
        ## Get the time and weights for the model.
        time = np.array([i/FPS for i in range(hockey_stick_angle.len())])
        angle_weights = np.array([1/hockey_stick_angle_error for _ in range(hockey_stick_angle.len())])

        ## Create the model, function is expected to be linear.
        angular_velocity_model = Model(linear)
        params = angular_velocity_model.make_params(a=1,b=0)

        ## Fit the data, the angular velocity should be the slope or "a" variable.
        result = angular_velocity_model.fit(hockey_stick_angle.to_numpy(), params, x=time, weights = angle_weights)
        slope_parameter = result.params.get("a")

        return slope_parameter.value, slope_parameter.stderr

    angular_velocity, angular_velocity_error = get_angular_velocity()

    # mean_hockey_stick_velocity = cast(float, hockey_stick_velocity.rolling_mean(window_size = ROLLING_MEAN_N).drop_nulls()[0])

    # hockey_stick_velocity_error = hockey_stick_velocity.rolling_std(window_size= ROLLING_MEAN_N).mean()
    # assert isinstance(hockey_stick_velocity_error, (float))

    ## Calculate rotation energy for the hockey stick.
    # angular_velocity = mean_hockey_stick_velocity/RADIUS
    # angular_velocity_relative_error = np.sqrt((hockey_stick_velocity_error/mean_hockey_stick_velocity)**2 + (RADIUS_ERROR/RADIUS)**2)
    #
    def get_hockey_stick_inertia_moment(stick_nr):
        return HOCKEY_STICK_INERTIA_MOMENTS[stick_nr-1], HOCKEY_STICK_INERTIA_MOMENT_ERRORS[stick_nr-1]

    inertia_moment_stick, inertia_moment_error = get_hockey_stick_inertia_moment(stick_nr)

    hockey_stick_rotation_energy = 0.5 * inertia_moment_stick * angular_velocity**2
    hockey_stick_rotation_energy_error = hockey_stick_rotation_energy * np.sqrt((2*angular_velocity_error/angular_velocity)**2 + (inertia_moment_error/inertia_moment_stick)**2)

    total_energy_after = ball_kinetic_energy+hockey_stick_rotation_energy + height_energy_gained
    total_energy_after_error = np.sqrt(
        ball_kinetic_energy_error**2
        + hockey_stick_rotation_energy_error**2
        + height_energy_gained_error**2
    )

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


## Linear function for fitting.
def linear(x,a,b):
    return a*x +b



def process_q_values(project_dir):
    ## Create dataframe to save results to.
    results = pl.DataFrame(
        schema={
            "q_mean": pl.Float64,
            "q_error": pl.Float64,
        }
    )
    ## Loop over the sticks and calculate the mean and the errors.
    for i in range(1,6):
        ## load data
        measurement_data = pl.read_csv(f"{project_dir}\\output\\output_meting_{i}.csv", separator = ",")

        q = measurement_data["q"]
        q_error = measurement_data["q_error"]

        ## get weights for weighted sum
        weights = 1/ q_error**2

        ## take the mean with a weighted sum so more accurate measurements contribute more.
        q_mean = (q*weights).sum() / weights.sum()  # pyright: ignore[reportOperatorIssue]
        q_mean_error = (1/weights.sum())**0.5   # pyright: ignore[reportOperatorIssue]

        ## save to dataframe usings .extend
        data_frame = pl.DataFrame({
            "q_mean": q_mean,
            "q_error": q_mean_error,
        })
        results.extend(data_frame)
    ## Write results to csv file.
    results.write_csv(f"{project_dir}\\output\\total_output.csv")
    ## Get indices for plotting
    indices = [i for i in range(1,6)]

    ## Create chart with error bars.
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
