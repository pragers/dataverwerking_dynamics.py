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
HOCKEY_STICK_INERTIA_MOMENTS = [0.16926240240330787,0.18367484458814398,0.17093826777363766,0.17563069081056104,0.1876969214769355]
HOCKEY_STICK_INERTIA_MOMENT_ERRORS = [0.005205745715940083, 0.005648079287620812, 0.005257176004501276,0.0054011864454451865, 0.005771533389105767]
HOCKEY_STICK_RADIUS = [0.421,0.42,0.409,0.405,0.406]
RADIUS_OFFSET = 0.02

## Constants
BALL_WEIGHTS = np.array([160e-3,157e-3,155e-3,161e-3,157e-3,155e-3]) # in kg
BALL_WEIGHT = BALL_WEIGHTS.mean()
BALL_WEIGHT_ERROR = 1e-3/np.sqrt(BALL_WEIGHTS.size) # in kg
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
RADIUS_ERROR = 2e-3 # in m

def main() -> None:
     convert_raw_data(os.getcwd())
     process_q_values(os.getcwd())
    # comp_vision_measurement_err(os.getcwd())

def convert_raw_data(project_dir):
    """
        Convert raw data to csv files with q, k_before, k_after, u_height and k_stick fields

        Parameters:
            project_dir: location of the project files.
    """
    ## We have 5 sticks, loop through them and get the q for all measurements. save them to csv file for later use.
    for metingnr in range(1,6):
        ## Get the measurements numbers
        start_measurement, amount_measurements = MEASUREMENTS[metingnr]

        ## check which index to start itterating from.
        start_measurement_index = MEASUREMENT_NRS.index(start_measurement)

        ## open csv to note results.
        with open(f"{project_dir}\\output\\output_meting_{metingnr}.csv", 'w+') as f:
            writer = csv.writer(f, lineterminator="\n")
            header = ["q","k_before","k_after","k_stick","q_error", "k_before_error", "k_after_error", "k_stick_error"]
            writer.writerow(header)

            ## loop over the measurements and calculate q, write the results to the csv for later processing.
            for i in range(amount_measurements):
                q, k_before, k_after, k_stick, q_error, k_before_error, k_after_error, k_stick_error = get_q(project_dir, metingnr, start_measurement_index + i, i+1)
                print(f"""Measurement {MEASUREMENT_NRS[start_measurement_index + i]}:
    Ratio of energy lost: {q}
    Kinetic energy before collision: {k_before} joule
    Kinetic energy after collision: {k_after} joule
    Hockey stick rotation energy: {k_stick} joule
                    """)
                writer.writerow([q, k_before, k_after, k_stick, q_error, k_before_error, k_after_error, k_stick_error])



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

    ## Get the ball's velocity using a linear fit. The slope of the fit is equal to the ball's velocity.
    ## The stderr of the fit is the error in the velocity.
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

    ## K = 1/2mv^2
    energy_before = 0.5 * BALL_WEIGHT * velocity ** 2
    energy_before_error = energy_before * np.sqrt((BALL_WEIGHT_ERROR/BALL_WEIGHT)**2 + (2*velocity_error/velocity)**2)


    ## Get the kinetic energy of the ball.
    ## We use a linear fit for the velocity and then the kinetic energy formula to determine the energy.
    ## Summing in quadrature of with same uncertainty doesnt change the uncertainty.
    ball_error = TRACKING_ACCURACY_METERS
    ball_velocity_x, ball_velocity_error_x = get_velocities(ball_data_after["x"] , ball_error)
    ball_velocity_y, ball_velocity_error_y = get_velocities(ball_data_after["y"] , ball_error)
    ball_velocity = np.sqrt(ball_velocity_x**2 + ball_velocity_y**2)
    ball_velocity_error = np.sqrt(
        (ball_velocity_x * ball_velocity_error_x)**2 +
        (ball_velocity_y * ball_velocity_error_y)**2
    ) / ball_velocity
    ## K = 1/2mv^2
    ball_kinetic_energy = 0.5 * BALL_WEIGHT * ball_velocity**2
    ball_kinetic_energy_error = ball_kinetic_energy * np.sqrt((BALL_WEIGHT_ERROR/BALL_WEIGHT)**2 + (2*ball_velocity_error/ball_velocity)**2)


    ## Calculate the total velocity of the hockey stick, by taking the pythagorean sum of the components.
    hockey_stick_place = ((hockey_data["X"]- hockey_data["X"][0])** 2 + (hockey_data["Y"] - hockey_data["Y"][0]) ** 2).sqrt()
    hockey_stick_place_error = np.sqrt(2) * TRACKING_ACCURACY_METERS
    ## Using the small angle aproximation.
    radius = HOCKEY_STICK_RADIUS[stick_nr-1] + RADIUS_OFFSET
    hockey_stick_angle = hockey_stick_place/radius
    hockey_stick_angle_error = (
        (hockey_stick_place_error / radius)**2
        + (hockey_stick_place * np.sqrt(2) * RADIUS_ERROR / radius**2)**2
    ).sqrt() ## we use this rewritten version of the absolute error to prevent division by zero error


    ## use a linear fit to get the angular velocity. The slope is the angular velocity.
    def get_angular_velocity():
        ## Get the time and weights for the model.
        time = np.array([i/FPS for i in range(hockey_stick_angle.len())])
        angle_weights = 1/ hockey_stick_angle_error.to_numpy()

        ## Create the model, function is expected to be linear.
        angular_velocity_model = Model(linear)
        params = angular_velocity_model.make_params(a=1,b=0)

        ## Fit the data, the angular velocity should be the slope or "a" variable.
        result = angular_velocity_model.fit(hockey_stick_angle.to_numpy(), params, x=time, weights = angle_weights)
        slope_parameter = result.params.get("a")

        return slope_parameter.value, slope_parameter.stderr

    angular_velocity, angular_velocity_error = get_angular_velocity()


    def get_hockey_stick_inertia_moment(stick_nr):
        return HOCKEY_STICK_INERTIA_MOMENTS[stick_nr-1], HOCKEY_STICK_INERTIA_MOMENT_ERRORS[stick_nr-1]

    inertia_moment_stick, inertia_moment_error = get_hockey_stick_inertia_moment(stick_nr)

    hockey_stick_rotation_energy = 0.5 * inertia_moment_stick * angular_velocity**2
    hockey_stick_rotation_energy_error = hockey_stick_rotation_energy * np.sqrt((2*angular_velocity_error/angular_velocity)**2 + (inertia_moment_error/inertia_moment_stick)**2)

    ## total energy after is the sum of all the different kinds of energy, the error could cancel partially so we add in quadrature.
    total_energy_after = ball_kinetic_energy+hockey_stick_rotation_energy
    total_energy_after_error = np.sqrt(
        ball_kinetic_energy_error**2
        + hockey_stick_rotation_energy_error**2
    )

    ## get the error for q.
    q_error = total_energy_after / energy_before * np.sqrt((total_energy_after_error/total_energy_after)**2 + (energy_before_error/energy_before)**2)

    return (
        1 - total_energy_after / energy_before,
        energy_before,
        ball_kinetic_energy,
        hockey_stick_rotation_energy,
        q_error,
        energy_before_error,
        ball_kinetic_energy_error,
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
    indices = [i+1 for i in range(5)]

    ## Create chart with error bars.
    chart = xy.errorbar_chart(
        xy.bar(
            indices,
            results["q_mean"].to_numpy(),
            color="#547BA5",
            opacity=0.9,
            corner_radius=4,
        ),
        xy.errorbar(
            indices,
            results["q_mean"].to_numpy(),
            yerr=results["q_error"].to_numpy(),
            color="#202020",
            width=1,
            cap_size = 0.2,
        ),
        xy.x_axis(label="Stick nr.",
            tick_count=6),
        xy.y_axis(
            label="Mean energy loss fraction q",
            bounds=(0, 1.1),
            domain = (0, 1.1),
            tick_count=5
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

    chart.to_png("output.png", width = 848, height =480)
    chart.to_svg("output.svg")





def comp_vision_measurement_err(project_dir) :
    ## Get the computer vision data and human data for 4707
    cv_data =  pl.read_csv(f"{project_dir}\\ball_data_before\\data_DSC_4707_before.csv", separator = ",")
    human_data =  pl.read_csv(f"{project_dir}\\Results2.csv", separator = ",")

    # Remove duplicate last frame and rescale.
    cv_data = cv_data.unique(maintain_order=True)
    cv_data *= PIXEL_SCALING_FACTOR

    # Get the rolling standard deviation to check the internal validity of the measurements.
    # Since there are basically no forces acting the standard deviation should reveal the uncertainty in the measurement.
    # Using a rolling std we dont compare the start speed to end speed, this minimizes the effect of any actual forces acting on the ball.
    print(cast(float,  human_data["X"].diff().mean())*12*3.6, cast(float,cv_data["x"].diff().mean()) *12 *3.6)
    print(cast(float,  human_data["X"].diff().rolling_std(window_size = 5).mean())*12*3.6, cast(float,cv_data["x"].diff().rolling_std(window_size = 5).mean()) *12 *3.6)

def get_maximum_comp_vision_std(project_dir,window_size):
    max_seen = 0
    seen_on= 0
    speeds = []
    for metingnr in range(1,6):
        start_measurement, amount_measurements = MEASUREMENTS[metingnr]
        start_measurement_index = MEASUREMENT_NRS.index(start_measurement)
        for i in range(amount_measurements):
            measurement_nr =MEASUREMENT_NRS[start_measurement_index + i]
            ball_data_before = pl.read_csv(f"{project_dir}\\ball_data_before\\data_DSC_{measurement_nr}_before.csv", separator = ",")
            ball_data_before = ball_data_before.unique(maintain_order=True)
            ball_data_before *= PIXEL_SCALING_FACTOR

            mean_std = ball_data_before["x"].diff().rolling_std(window_size=window_size).mean()
            avg_speed = cast(float,(ball_data_before["x"].diff(n=DIFF_N)/DIFF_N).mean())*12 * 3.6;
            speeds.append(avg_speed)
            ##print(measurement_nr, mean_std)
            if isinstance(mean_std,float) and mean_std> max_seen :
                max_seen = mean_std
                seen_on = measurement_nr
    print(*[f"{i}: {x:.2f}km/h\n" for i,x in enumerate(speeds)])
    return max_seen, seen_on
