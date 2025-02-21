""" Optimization initial code  """

import numpy as np
from scipy.optimize import shgo, differential_evolution, minimize
from multiprocessing import Pool, cpu_count
from my_project.simulation import simulate_battery
from . import config

def compute_metrics(solar_size, wind_size, battery_size, df, config_list, colocation_option):
    solar_power = df['Solar Power Per MW'].values
    wind_power = df['Wind Power Per MW'].values
    demand = df['Data center Actual Demand'].values
    is_peak = df['is_peak'].values

    Pmax = battery_size / config.BATTERY_BACKUP_HOURS
    SOC = 0.4  # initial state-of-charge (40%)
    total_discharge = total_charge = total_excess_exported = 0
    total_peak_deficit = total_energy_generated = 0

    for i in range(len(df)):
        solar_gen = solar_power[i] * solar_size if 'Solar' in config_list else 0
        wind_gen = wind_power[i] * wind_size if 'Wind' in config_list else 0
        total_gen = solar_gen + wind_gen

        excess_energy = max(0, total_gen - demand[i])
        deficit = max(0, demand[i] - total_gen)

        charge = min(excess_energy, Pmax, battery_size * (1 - SOC))
        SOC += (charge * config.BATTERY_CHARGING_EFFICIENCY) / battery_size

        discharge = 0
        if is_peak[i] == 1:
            discharge = min(deficit, Pmax, SOC * battery_size)
            SOC -= discharge / (battery_size * config.BATTERY_DISCHARGING_EFFICIENCY)
            total_peak_deficit += deficit
            deficit -= discharge

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
    func, x = args
    return func(x)

def objective_function(x, df, config_list, colocation_option):
    solar_size, wind_size, battery_size = x
    RE_util, batt_util, peak_met, annual_dc_demand_meet = compute_metrics(solar_size, wind_size, battery_size, df, config_list, colocation_option)

    penalty = 0
    if RE_util < 95:
        penalty += 1000 * (95 - RE_util)
    if batt_util < 80:
        penalty += 1000 * (80 - batt_util)
    if peak_met < 70:
        penalty += 1000 * (70 - peak_met)
    if colocation_option in [2, 3, 4] and annual_dc_demand_meet < 80:
        penalty += 1000 * (80 - annual_dc_demand_meet)

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
                    objective_function, bounds,
                    args=(df, config_list, colocation_option),
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
