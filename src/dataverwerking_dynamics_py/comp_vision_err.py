from dataverwerking_dynamics_py import MEASUREMENT_NRS, MEASUREMENTS, PIXEL_SCALING_FACTOR
import polars as pl

def comp_vision_measurement_err(project_dir) :
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
