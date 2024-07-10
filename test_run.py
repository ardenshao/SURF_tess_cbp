import sys
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
import pandas as pd
import os
import warnings
import math
from sklearn.linear_model import LinearRegression
from lightkurve import search_lightcurve
from bokeh.plotting import figure, output_file, show, save
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool, Range1d
warnings.filterwarnings('ignore')


def filter_data_around_transits(df, transit_times, window=2):
    """
    Filters the DataFrame to include only rows where the jd value falls within the window around each transit time.
    
    Parameters:
    - df: DataFrame containing the time series data.
    - transit_times: Array of transit times.
    - window: Number of days around each transit time to consider for filtering.
    
    Returns:
    - Filtered DataFrame.
    """
    mask = pd.Series(False, index=df.index)
    for transit in transit_times:
        lower_bound = transit - window
        upper_bound = transit + window
        mask |= (df['jd'] >= lower_bound) & (df['jd'] <= upper_bound)
    return df[mask]


def filter_data(df, quality, flux_threshold):
    """Filter data based on quality and flux."""
    return df[(df['quality'] == quality) & (df['flux'] >= flux_threshold)]


def detrend_data(df):
    """Fit linear regression model to predict detrended data."""
    time = df['jd'].values.reshape(-1, 1)
    flux = df['flux'].values
    
    model = LinearRegression()
    model.fit(time, flux)
    trend = model.predict(time)
    
    detrended_flux = flux - trend + 1

    df['flux'] = detrended_flux

    return df
    

def add_tess_points(df):
    """
    Augments a given DataFrame with additional rows to ensure that consecutive 'jd' values are at least 0.1 units apart.
    """
    df['jd_diff'] = df['jd'].diff()
    
    filtered_rows = df[df['jd_diff'] > 0.1]
    
    # for each group of consecutive 'jd' values that are less than or equal to 0.1 apart,
    # create a new row with 'jd' incremented by 0.1 and 'flux' set to 1.0
    new_rows = []
    for _, group in filtered_rows.groupby(filtered_rows.index // 10):
        new_row = {'jd': group.iloc[0]['jd'] + 0.1 * len(group), 'flux': 1.0}
        new_rows.append(new_row)
    
    combined_rows = pd.concat([df.drop(columns=['jd_diff']), pd.DataFrame(new_rows)])
    
    final_df = combined_rows.drop_duplicates(subset='jd', keep='first')
    
    final_df.reset_index(drop=True, inplace=True)
    return final_df


def chi_squared(tess_df, comparison_df, period, transit):
    folded_tess = lk.LightCurve(time=tess_df['jd'], 
                             flux=tess_df['flux']).fold(period=period, epoch_time=transit)
    folded_comparison = lk.LightCurve(time=comparison_df['jd'], 
                               flux=comparison_df['flux']).fold(period=period, epoch_time=transit)

    folded_tess_df = pd.DataFrame({'jd': folded_tess.time.value, 'flux': folded_tess.flux.value})
    folded_comparison_df = pd.DataFrame({'jd': folded_comparison.time.value, 'flux': folded_comparison.flux.value, 
                                         'flux_err': comparison_df['flux_err']}) # we already calculated and added 'flux_err' previously

    # calculate sd of comparison dataset's flux errors


    comparison_err = np.std(folded_comparison_df['flux'])
    chi_squared_sum = 0
    
    for _, row in folded_comparison_df.iterrows():
        comparison_phase = row['jd']
        comparison_flux = row['flux']
        
        differences = np.abs(folded_tess_df['jd'] - comparison_phase)
    
        closest_idx = differences.argmin()
        
        closest_tess_flux = folded_tess_df.loc[closest_idx, 'flux']
        
        chi_squared = ((comparison_flux - closest_tess_flux) / comparison_err)**2
        chi_squared_sum += chi_squared
        
    return chi_squared_sum


def automate_periodogram(tid, primary_transit, secondary_transit, min_period, max_period):
    """
    Generates a primary periodogram for a given TESS ID, incorporating WASP and ASAS-SN data.

    Parameters:
    - tid (str): TESS ID of the target.
    - primary_transit (float): JD of primary transit epoch.
    - secondary_transit (float): JD of secondary transit epoch.
    - min_period (float): Minimum period for periodogram.
    - max_period (float): Maximum period for periodogram.

    Displays a plot of the best predicted period by lowest chi-squared.
    """
    # get the TESS data
    try:
        tid_str = str(tid)
        search_result = search_lightcurve(tid_str, mission='TESS')
        tess_lc = search_result.download().remove_nans().normalize()
    except Exception as e:
        print(f"Error processing {tid} ({type(e).__name__}): {e}")

    # detrend method requires a df, so convert tess_lc to a dataframe
    tess_df = pd.DataFrame({
        'jd': tess_lc.time.value,
        'flux': tess_lc.flux.value
    })
    tess_df = detrend_data(tess_df)

    # also add points to tess
    tess_df = add_tess_points(tess_df)

    base_dir = f'output/{tid}'
    asassn_df = None
    
    for item in os.listdir(base_dir):
        
        # attempt to open the file starting with "asas_sn_id..."
        for filename in os.listdir(base_dir):
            if filename.startswith("asas_sn_id"):
                with open(os.path.join(base_dir, filename), 'r') as file:
                    
                    # read the file content into a DataFrame
                    asassn_df = pd.read_csv(file, skiprows=1)

    # step 1: filter
    asassn_df = filter_data(asassn_df, 'G', 0)

    # step 2: normalize
    asassn_median = asassn_df['flux'].median()
    
    asassn_data = {
        'jd': asassn_df['jd'] - 2457000,
        'flux': asassn_df['flux'] / asassn_median,
        'flux_err': asassn_df['flux_err'],
        'phot_filter': asassn_df['phot_filter']
    }
    asassn_df = pd.DataFrame(asassn_data)

    # step 3: detrend
    filters = ['g', 'V']
    all_detrended_data = []
    
    for filter_type in filters:
        filtered_data = asassn_df[asassn_df['phot_filter'] == filter_type]
    
        all_detrended_data.append(detrend_data(filtered_data))

    detrended_asassn_df = pd.concat(all_detrended_data)

    asassn_detrended = {
        'jd': detrended_asassn_df['jd'],
        'flux': detrended_asassn_df['flux'],
        'flux_err': asassn_df['flux_err'] / (detrended_asassn_df['flux']).median(),
    }
    
    asassn_df = pd.DataFrame(asassn_detrended)


    for _ in os.listdir(base_dir):
        
        # attempt to open the file starting with "1SWASP"
        for filename in os.listdir(base_dir):
            if filename.startswith("1SWASP"):
                wasp_filepath = os.path.join(base_dir, filename)
    
    # obtain the wasp data
    with open(wasp_filepath, 'r') as my_file:
        for _ in range(22):
            next(my_file)
        
        data = []
        for line in my_file:
            fields = line.split()
            tamflux2_value = float(fields[3])
            tamflux2_err_value = float(fields[4])
            hjd_value = float(fields[9])
            data.append({'jd': hjd_value, 'flux': tamflux2_value, 'flux_err': tamflux2_err_value})
    
    wasp_df = pd.DataFrame(data)
    
    # normalize
    wasp_median = wasp_df['flux'].median()
    
    wasp_data = {
        'jd': wasp_df['jd'] - 2457000,
        'flux': wasp_df['flux'] / wasp_median,
        'flux_err': wasp_df['flux_err'] / wasp_median,
    }
    wasp_df = pd.DataFrame(wasp_data)

    # detrend wasp
    wasp_df = detrend_data(wasp_df)

    # add labels
    tess_df['dataset_marker'] = 'TESS'
    asassn_df['dataset_marker'] = 'ASAS-SN'
    wasp_df['dataset_marker'] = 'SWASP'
  
    # combine the data + filter 
    combined_df = pd.concat([tess_df, asassn_df, wasp_df])
    combined_df.to_csv(f'output/{tid}/combined_data.csv', index=False)

    # period calculation using minimizing chi-squared method:
    periods = np.arange(min_period, max_period, 0.0001)

    best_predicted_periods = []
    transits = [primary_transit, secondary_transit]
    fig, axs = plt.subplots(2, figsize=(12, 10))  # Adjust figsize as needed

    for i, transit in enumerate(transits):
        chi_squared_values = []
    
        for period in periods:
            transit_times = np.concatenate((np.arange(transit, np.min(combined_df["jd"]) - period, -period), 
                                            np.arange(transit + period, np.max(combined_df["jd"]) + period, period)))
    
            filtered_tess_df = filter_data_around_transits(tess_df, transit_times)
            filtered_asassn_df = filter_data_around_transits(asassn_df, transit_times)
            filtered_wasp_df = filter_data_around_transits(wasp_df, transit_times)
            
            chi_squared_sum = chi_squared(filtered_tess_df, filtered_asassn_df, period, transit) + chi_squared(filtered_tess_df, filtered_wasp_df, period, transit)
    
            chi_squared_values.append(chi_squared_sum)
    
        best_predicted_period = periods[np.argmin(chi_squared_values)]
        best_predicted_periods.append(best_predicted_period)

        axs[i].plot(periods, chi_squared_values)
        index_of_best_period = list(periods).index(best_predicted_period)
        axs[i].plot(best_predicted_period, chi_squared_values[index_of_best_period], 'ro')
        axs[i].set_xlabel('Period')
        axs[i].set_ylabel('Chi-Squared')
        axs[i].set_title(f'Best Predicted Period: {best_predicted_period:.5f}')
        axs[i].grid()

    plt.savefig(f'output/{tid}/periodogram.png')
    plt.tight_layout()  # Adjust subplot parameters to give specified padding
    plt.show()

    # write predicted periods to csv file
    tess_eb_df = pd.read_csv('tess_ebs_data.csv')
    
    filtered_row = tess_eb_df[tess_eb_df['tid'] == tid]
    
    filtered_row.loc[:, 'predicted_primary_period'] = best_predicted_periods[0] 
    filtered_row.loc[:, 'predicted_secondary_period'] = best_predicted_periods[1]
    tess_eb_df.update(filtered_row)
    tess_eb_df.to_csv('tess_ebs_data.csv', index=False)
    
    return best_predicted_periods


def detect_precession_bokeh(tid, fold_periods, primary_transit, secondary_transit):
    
    df = pd.read_csv(f'output/{tid}/combined_data.csv')
    marker_to_color = {'TESS': 'blue', 'ASAS-SN': 'green', 'SWASP': 'red'}
    marker_to_shape = {'TESS': 'v', 'ASAS-SN': 'o', 'SWASP': 'x'}  # Define shapes here
    
    plots = []
    transits = [primary_transit, secondary_transit]
    
    # Calculate dataset proportions
    dataset_counts = df['dataset_marker'].value_counts().sort_index()
    total_count = len(df)
    dataset_proportions = 1 - dataset_counts / total_count
    
    for period in fold_periods:
        for transit in transits:
            p = figure(title=f'Folding with period of {period:.5f} d for transit at {transit} JD', 
                       x_axis_label='Phase', y_axis_label='Flux', width=600, height=600)
            
            for marker, color in marker_to_color.items():
                marker_df = df[df['dataset_marker'] == marker]
                shape = marker_to_shape.get(marker, 'o')  
                folded_lc = LightCurve(time=marker_df['jd'], flux=marker_df['flux']).fold(period=period, epoch_time=transit)
                
                # Set alpha based on dataset proportion
                alpha = dataset_proportions.loc[marker]  # Assuming the index is the marker name
                
                source = ColumnDataSource(data=dict(phase=folded_lc.time.value, flux=folded_lc.flux, marker=[marker]*len(folded_lc.time)))
                p.scatter('phase', 'flux', source=source, size=3, color=color, alpha=alpha, legend_label=marker, marker=shape)
            
            p.legend.title = 'Datasets'
            p.legend.location = 'top_left'
            p.add_tools(HoverTool(tooltips=[("Phase", "@phase"), ("Flux", "@flux"), ("Dataset", "@marker")]))
            p.x_range = Range1d(-0.5, 0.5)
            p.xgrid.grid_line_color = None
            p.ygrid.grid_line_color = None
            plots.append(p)
    
    grid = gridplot([plots[i:i+2] for i in range(0, len(plots), 2)])
    output_file(f'output/{tid}/folded.html')
    save(grid)
    show(grid)


def round_to_decimal_place(num, decimal_places=3):
    """Round a number to a specified number of decimal places."""
    factor = 10 ** decimal_places
    
    lower_num = math.floor(num * factor) / factor
    upper_num = math.ceil(num * factor) / factor
    
    return lower_num, upper_num


def get_row_by_tid(tid):
    filtered_df = df[df['tid'] == tid]
    return filtered_df.iloc[0]


def check_files_exist(files, prefixes):
    return all(any(file.startswith(prefix) for prefix in prefixes) for file in files)


def extract_index(string):
    reversed_string = string[::-1]
    empty_string = ''
    index = 0
    while True:
        is_number = reversed_string[4+index].isdigit()
        if is_number == True:
            empty_string += reversed_string[4+index]
        else:
            return int(empty_string[::-1])
        index += 1


df = pd.read_csv('tess_ebs_data.csv')
base_dir = "output/"

with open("no_kelt_tids.txt", "r") as file:
    no_kelt_tids = [line.strip() for line in file]

items = os.listdir(base_dir)
prefixes = ["1SWASP", "asas_sn_id"]

my_string = sys.argv[1]
run_index = extract_index(my_string)

item = items[run_index]

item_path = os.path.join(base_dir, item)

if os.path.isdir(item_path) and item in no_kelt_tids:
    files_in_directory = os.listdir(item_path)
    
    if check_files_exist(files_in_directory, prefixes):
        tid = item  # 'item' is the tid here
        row = get_row_by_tid(tid)

        print(f"Processing tid: {tid}")

        # getting info about the object
        ra = row['ra']
        dec = row['dec']
        primary_transit = row['primary_transit']
        secondary_transit = row['secondary_transit']
        period = row['period']

        min_period, max_period = round_to_decimal_place(period, decimal_places=2)
        min_period -= 0.01
        max_period += 0.01

        fold_periods = automate_periodogram(tid, primary_transit, secondary_transit, 
                                            min_period, max_period)

        detect_precession_bokeh(tid, fold_periods, primary_transit, secondary_transit)
