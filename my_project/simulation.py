import pandas as pd
from . import config

def simulate_battery(df, solar_size, wind_size, battery_size, colocation_option, re_utilization, output_path, contract_demand):
    # Local variables for cost and energy accumulation.
    total_solar_cost = 0.0
    total_wind_cost = 0.0
    total_market_cost = 0.0
    total_grid_cost = 0.0
    total_export_revenue = 0.0
    total_battery_cost = 0.0

    x = config.TRANSMISSION_COST if colocation_option in [4, 5] else 0
    y = config.TRANSMISSION_COST if colocation_option in [2, 4] else 0

    solar_cost_per_MWh = config.SOLAR_PPA + x
    wind_cost_per_MWh = config.WIND_PPA + y

    # Battery cost parameters
    Pmax = battery_size / config.BATTERY_BACKUP_HOURS
    SOC = 0.4  # Initial state-of-charge
    battery_fixed_cost = config.BATTERY_FIXED_COST_FACTOR * (battery_size / config.BATTERY_BACKUP_HOURS)
    storage_cost_per_MWh = config.BATTERY_FIXED_COST_FACTOR / (365 * config.BATTERY_BACKUP_HOURS)
    battery_cost_per_MWh = storage_cost_per_MWh + solar_cost_per_MWh
    total_battery_cost = battery_fixed_cost

    # Convert RE utilization percentage into a fraction.
    re_util_frac = re_utilization / 100.0

    results = []
    max_grid_demand_per_month = {}  # Dictionary to track max grid demand per month
    max_grid_cost_per_day = {}  # Dictionary to track max grid cost per day

    for i, row in df.iterrows():
        solar_gen = row['Solar Power Per MW'] * solar_size
        wind_gen = row['Wind Power Per MW'] * wind_size
        total_gen = solar_gen + wind_gen
        demand = row['Data center Actual Demand']
        market_price = row['MCP (Rs/MWh)'] + config.TRANSMISSION_COST
        selling_price = row['MCP (Rs/MWh)']

        # Determine grid cost based on the hour of the day
        hour = row['Hour']
        if 6 <= hour <= 8 or 18 <= hour <= 21:
            grid_cost_val = 9330
        elif 9 <= hour <= 17:
            grid_cost_val = 8330
        else:
            grid_cost_val = 7330

        # Track the maximum grid cost value for each day
        day = row['Day']
        if day not in max_grid_cost_per_day:
            max_grid_cost_per_day[day] = grid_cost_val
        else:
            max_grid_cost_per_day[day] = max(max_grid_cost_per_day[day], grid_cost_val)

        peak = row['is_peak']
        month = row['Month']  # Extract month

        excess_energy = max(0, total_gen - demand)
        deficit = max(0, demand - total_gen)

        # Calculate solar and wind costs
        solar_cost = (solar_gen * re_util_frac * solar_cost_per_MWh) + (solar_gen * (1 - re_util_frac) * config.FALLBACK_SOLAR_COST)
        wind_cost = (wind_gen * re_util_frac * wind_cost_per_MWh) + (wind_gen * (1 - re_util_frac) * config.FALLBACK_WIND_COST)
        total_solar_cost += solar_cost
        total_wind_cost += wind_cost

        # Battery charging logic
        charge = min(excess_energy, Pmax, (battery_size * (1 - SOC)))
        SOC += (charge * config.BATTERY_CHARGING_EFFICIENCY) / battery_size

        # Calculate exported energy
        exported = max(0, excess_energy - charge)  # Whatever remains after charging is exported

        # Calculate recovery revenue
        recovery_revenue = exported * selling_price
        total_export_revenue += recovery_revenue  # Accumulate the revenue

      #  # Battery discharging
        discharge = 0.0
        if peak == 1:
            discharge = min(deficit, Pmax, SOC * battery_size)
            SOC -= discharge / (battery_size * config.BATTERY_DISCHARGING_EFFICIENCY)
            deficit -= discharge
        # discharge = 0.0
        # battery_used = 0
        # if battery_cost_per_MWh < min(market_price, grid_cost_val) and deficit >0:
        #     battery_used = 1

        # if battery_used == 1:
        #     discharge = min(deficit, Pmax, SOC * battery_size)
        #     SOC -= discharge / (battery_size * config.BATTERY_DISCHARGING_EFFICIENCY)
        #     deficit -= discharge

### Remaining energy procurement logic

        market_energy, grid_energy = 0.0, 0.0
        if deficit > 0:
            if market_price < grid_cost_val:
                market_energy = deficit
                total_market_cost += market_energy * market_price
            else:
                grid_energy = deficit
                total_grid_cost += grid_energy * grid_cost_val
                max_grid_demand_per_month[month] = max(max_grid_demand_per_month.get(month, 0), grid_energy)

        results.append({
            'Day': day,
            'Hour': hour,
            'Solar Generation (MW)': solar_gen,
            'Wind Generation (MW)': wind_gen,
            'Total Generation (MW)': total_gen,
            'Data center Actual Demand': demand,
            'Excess Energy (MW)': excess_energy,
            'Deficit Energy (MW)': deficit,
            'is_peak': peak,
            'grid_peak': row['grid_peak'],
            'SOC': SOC,
          #  'Battery used': battery_used,
            'Battery Charge (MW)': charge,
            'Market Price ': market_price,
            'Grid cost': grid_cost_val,
            'Battery_cost_per_MWh': battery_cost_per_MWh,
            'Battery Discharge (MW)': discharge,
            'Excess Exported (MW)': exported,
            'Market Energy (MW)': market_energy,
            'Grid Energy (MW)': grid_energy,
            'Grid Cost (Rs)': grid_energy * grid_cost_val if grid_energy > 0 else 0.0,
            'Market Cost (Rs)': market_energy * market_price if market_energy > 0 else 0.0,
        })

    # Calculate demand charges
    demand_charge_total = 0.0
    for month, max_demand in max_grid_demand_per_month.items():
        max_demand_MVA = max_demand / config.POWER_FACTOR  # Convert MW to MVA
        effective_demand = max(0.85 * contract_demand, max_demand_MVA)
        demand_charge = effective_demand * config.DEMAND_CHARGE_PER_MVA
        demand_charge_total += demand_charge

    # Calculate total grid cost if 100% of the demand is met from the grid
    total_energy_charge_full_grid = 0.0
    for i, row in df.iterrows():
        demand = row['Data center Actual Demand']
        hour = row['Hour']
        if 6 <= hour <= 8 or 18 <= hour <= 21:
            grid_cost_val = 9330
        elif 9 <= hour <= 17:
            grid_cost_val = 8330
        else:
            grid_cost_val = 7330
        total_energy_charge_full_grid += demand * grid_cost_val

    total_energy_charge_full_grid /= 1e7  # Convert to Cr (Rs)

    # Calculate the demand charge for full grid scenario
    max_grid_demand_full_grid_per_month = {}
    for i, row in df.iterrows():
        demand = row['Data center Actual Demand']
        month = row['Month']
        max_grid_demand_full_grid_per_month[month] = max(max_grid_demand_full_grid_per_month.get(month, 0), demand)

    demand_charge_total_full_grid = 0.0
    for month, max_demand in max_grid_demand_full_grid_per_month.items():
        max_demand_MVA = max_demand / config.POWER_FACTOR  # Convert MW to MVA
        effective_demand = max(0.85 * contract_demand, max_demand_MVA)
        demand_charge = effective_demand * config.DEMAND_CHARGE_PER_MVA
        demand_charge_total_full_grid += demand_charge

    total_grid_cost_full_grid = demand_charge_total_full_grid + total_energy_charge_full_grid
    grid_avg_cost_per_unit = (total_grid_cost_full_grid / (df['Data center Actual Demand'].sum() * 1000)) * 10**7

    # Create a dictionary with cost components
    cost_dict = {
        'total_solar_cost': total_solar_cost / 1e7,  # Convert to Cr (Rs)
        'total_wind_cost': total_wind_cost / 1e7,
        'total_battery_cost': total_battery_cost / 1e7,
        'total_market_cost': total_market_cost / 1e7,
        'total_grid_cost': total_grid_cost / 1e7,
        'battery_cost_per_MWh': battery_cost_per_MWh ,
        'total_export_revenue': total_export_revenue / 1e7,
        'total_demand_charge': demand_charge_total / 1e7,  # Convert to Cr (Rs)
        'total_cost_with_demand_charge_with_selling': (total_solar_cost + total_wind_cost + total_battery_cost + total_market_cost + total_grid_cost + demand_charge_total - total_export_revenue) / 1e7,
        'total_cost_with_demand_charge_without_selling': (total_solar_cost + total_wind_cost + total_battery_cost + total_market_cost + total_grid_cost + demand_charge_total) / 1e7,
        'total_grid_cost_full_grid': total_grid_cost_full_grid,
        'total_energy_charge_full_grid': total_energy_charge_full_grid,
        'total_demand_charge_full_grid': demand_charge_total_full_grid / 1e7,
        'grid_avg_cost_per_unit': grid_avg_cost_per_unit
    }

    # Convert results to DataFrame and save to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_path, index=False)

    # Return the maximum grid cost value for the day
    max_grid_cost_val = max(max_grid_cost_per_day.values())

    return results_df, cost_dict, max_grid_cost_val
