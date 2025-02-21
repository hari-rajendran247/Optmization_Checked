""" Code for Discharging strategy when is_peak=1  ie the battery
discharges only during peak period.  is_peak =1 during the hours [18,19,20 & 21].
But the grid is having peak at hours =[7,8,9,10,18,19,20,21].
The grid peak is not considered in the code"""

import numpy as np
from scipy.optimize import shgo, differential_evolution, minimize
from multiprocessing import Pool, cpu_count
from my_project.simulation import simulate_battery
from . import config

def compute_metrics(solar_size, wind_size, battery_size, df, config_list, colocation_option):
    solar_power = df['Solar Power Per MW'].values
    wind_power = df['Wind Power Per MW'].values
    demand = df['Data center Actual Demand'].values
    market_price = df['MCP (Rs/MWh)'].values + config.TRANSMISSION_COST
    hour = df['Hour'].values
    is_peak = df['is_peak'].values

    Pmax = battery_size / config.BATTERY_BACKUP_HOURS
    SOC = 0.4  # initial state-of-charge (40%)
    total_discharge = total_charge = total_excess_exported = 0
    total_peak_deficit = total_energy_generated = 0
    total_solar_cost = total_wind_cost = total_battery_cost = 0
    total_grid_cost = total_market_cost = 0
    max_grid_demand_per_month = {i: 0 for i in range(1, 13)}

    # Calculate solar and wind costs based on colocation_option
    x = config.TRANSMISSION_COST if colocation_option in [4, 5] else 0
    y = config.TRANSMISSION_COST if colocation_option in [2, 4] else 0

    solar_cost_per_MWh = config.SOLAR_PPA + x
    wind_cost_per_MWh = config.WIND_PPA + y

    # Calculate battery cost per MWh
    battery_cost_per_MWh = config.BATTERY_FIXED_COST_FACTOR / (365 * config.BATTERY_BACKUP_HOURS) + solar_cost_per_MWh

    for i in range(len(df)):
        solar_gen = solar_power[i] * solar_size if 'Solar' in config_list else 0
        wind_gen = wind_power[i] * wind_size if 'Wind' in config_list else 0
        total_gen = solar_gen + wind_gen

        excess_energy = max(0, total_gen - demand[i])
        deficit = max(0, demand[i] - total_gen)

        # Determine grid cost based on the hour of the day
        if 6 <= hour[i] <= 8 or 18 <= hour[i] <= 21:
            grid_cost_val = 9330
        elif 9 <= hour[i] <= 17:
            grid_cost_val = 8330
        else:
            grid_cost_val = 7330

        charge = min(excess_energy, Pmax, battery_size * (1 - SOC))
        SOC += (charge * config.BATTERY_CHARGING_EFFICIENCY) / battery_size

        discharge = 0
        if is_peak[i] == 1:
            discharge = min(deficit, Pmax, SOC * battery_size)
            SOC -= discharge / (battery_size * config.BATTERY_DISCHARGING_EFFICIENCY)
            total_peak_deficit += deficit
            deficit -= discharge
        else:
            # Energy procurement if deficit remains
            market_energy, grid_energy = 0.0, 0.0
            if deficit > 0:
                if market_price[i] < grid_cost_val:
                    market_energy = deficit
                    total_market_cost += market_energy * market_price[i]
                else:
                    grid_energy = deficit
                    total_grid_cost += grid_energy * grid_cost_val
                    month = (df.index[i].month if 'Month' not in df.columns else df['Month'][i])
                    max_grid_demand_per_month[month] = max(max_grid_demand_per_month.get(month, 0), grid_energy)



        exported = max(0, excess_energy - charge)
        total_discharge += discharge
        total_charge += charge
        total_excess_exported += exported
        total_energy_generated += total_gen

    RE_utilization = (total_energy_generated - total_excess_exported) / total_energy_generated * 100
    battery_utilization = (total_discharge / (battery_size * 365 * config.RTE)) * 100
    peak_demand_met = (total_discharge / total_peak_deficit) * 100 if total_peak_deficit > 0 else 100

    # Calculate Annual DC demand meet %
    total_demand = demand.sum()
    annual_dc_demand_meet = ((total_energy_generated - total_excess_exported) / total_demand) * 100

    return RE_utilization, battery_utilization, peak_demand_met, annual_dc_demand_meet

def objective_function_wrapper(args):
    return objective_function(*args)

def objective_function(x, df, config_list, colocation_option):
    solar_size, wind_size, battery_size = x
    RE_util, batt_util, peak_met, annual_dc_demand_meet = compute_metrics(solar_size, wind_size, battery_size, df, config_list, colocation_option)

    penalty = 0
    if RE_util < 95:
        penalty += 1000 * (95 - RE_util)
    if batt_util < 85:
        penalty += 1000 * (85 - batt_util)
    if peak_met < 70:
        penalty += 1000 * (70 - peak_met)
    if colocation_option in [2, 3, 4] and annual_dc_demand_meet < 70:
        penalty += 1000 * (70 - annual_dc_demand_meet)

    if penalty > 0:
        return penalty

    df_sim = simulate_battery(df.copy(), solar_size, wind_size, battery_size, colocation_option, RE_util)
    total_cost_without_exports = (
        df_sim['total_solar_cost'].sum() + df_sim['total_wind_cost'].sum() + df_sim['total_battery_cost'].sum() +
        df_sim['total_grid_cost'].sum() + df_sim['total_market_cost'].sum()
    )
    total_demand = df_sim['Data center Actual Demand'].sum()
    cost_without_selling = total_cost_without_exports / (total_demand * 1000)
    return cost_without_selling

def optimize_system(df, config_list, scaling_factor, colocation_option=1, config_choice='1'):
    bounds = [
        (10 * scaling_factor, 270 * scaling_factor),
        (10 * scaling_factor, 270 * scaling_factor),
        (10 * scaling_factor, 440 * scaling_factor)
    ]

    if 'Solar' not in config_list:
        bounds[0] = (0, 0)
    if 'Wind' not in config_list:
        bounds[1] = (0, 0)
    if 'Battery' not in config_list:
        bounds[2] = (0, 0)

    def run_optimization(bounds):
        # Step 1: Global Optimization using SHGO or GA
        if config_choice in ['1', '5']:
            global_result = differential_evolution(
                objective_function, bounds,
                args=(df, config_list, colocation_option),
                maxiter=config.OPT_MAXITER, popsize=config.OPT_POPSIZE, tol=config.OPT_TOL,
                updating='deferred' if cpu_count() > 1 else 'immediate',
                workers=cpu_count()
            )
        else:
            with Pool(cpu_count(), maxtasksperchild=1000) as pool:
                global_result = shgo(
                    objective_function_wrapper, bounds,
                    args=((df, config_list, colocation_option),),
                    n=1000,
                    iters=20,
                    sampling_method='sobol',
                    options={'ftol': 1e-7, 'xtol': 1e-7}
                )

        # Step 2: Local Optimization using L-BFGS-B
        local_result = minimize(
            objective_function, global_result.x,
            args=(df, config_list, colocation_option),
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': False}
        )

        return local_result

    max_expansions = 3  # Prevent infinite loops
    expansion_count = 0
    prev_objective_value = float('inf')

    while expansion_count < max_expansions:
        result = run_optimization(bounds)
        x_opt = result.x
        obj_value = result.fun  # Current best objective function value

        # Stop expanding if the objective function is not improving significantly
        if abs(prev_objective_value - obj_value) < 1e-3:
            break
        prev_objective_value = obj_value

        # Check if any variable is at the bounds
        bounds_expanded = False
        new_bounds = []
        for i, (low, high) in enumerate(bounds):
            if x_opt[i] <= low + 1e-3:  # Close to lower bound
                new_low = max(0, low * 0.8)  # Reduce lower bound by 20%
                new_bounds.append((new_low, high))
                bounds_expanded = True
            elif x_opt[i] >= high - 1e-3:  # Close to upper bound
                new_high = high * 1.2  # Expand upper bound by 20%
                new_bounds.append((low, new_high))
                bounds_expanded = True
            else:
                new_bounds.append((low, high))

        if not bounds_expanded:
            break  # Stop expanding if no bounds were hit

        bounds = new_bounds
        expansion_count += 1  # Increment expansion counter

    return {
        'solar_size': x_opt[0],
        'wind_size': x_opt[1],
        'battery_size': x_opt[2],
        'cost_without_selling': obj_value
    }
