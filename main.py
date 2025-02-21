import time
import pandas as pd
from my_project import config, optimization, simulation, utils

def main():
    start_time = time.time()

    # Loading the data with per MW Demand, Solar, and Wind profile
    data_file = 'data/ip_data_karnataka.xlsx'
    df = utils.load_data(data_file)

    # Taking input from user for the required demand
    try:
        user_demand = float(input("Enter desired total demand (MW) (e.g. 150, 200, etc.): "))
    except ValueError:
        print("Invalid input for demand.")
        return
    scaling_factor = user_demand / config.BASELINE_DEMAND  # for adjusting the bounds dynamically

    # Scale the per-MW demand profile
    df['Data center Actual Demand'] *= user_demand

    # Ensure 'Month' column is present
    if 'Month' not in df.columns:
        df['Month'] = pd.to_datetime(df['Day'], format='%Y-%m-%d').dt.month

    # User Input: Configuration Selection
    configurations = {
        '1': ('Solar (Colocated) + Battery', ['Solar', 'Battery'], 1),
        '2': ('Solar (Colocated) + Wind (Non-Colocated) + Battery', ['Solar', 'Wind', 'Battery'], 2),
        '3': ('Solar (Colocated) + Wind (Colocated) + Battery', ['Solar', 'Wind', 'Battery'], 3),
        '4': ('Solar (Non-Colocated) + Wind (Non-Colocated) + Battery', ['Solar', 'Wind', 'Battery'], 4),
        '5': ('Solar (Non-Colocated) + Battery', ['Solar', 'Battery'], 5)
    }
    results = []


    for config_choice, (config_name, config_list, colocation_choice) in configurations.items():
        print(f"\nRunning optimization for {config_name} configuration...")

        opt_result = optimization.optimize_system(df, config_list, scaling_factor, colocation_choice)

        # Extract values correctly from dictionary
        solar_size = opt_result['solar_size']
        wind_size = opt_result['wind_size']
        battery_size = opt_result['battery_size']

        RE_util, batt_util, peak_met, annual_dc_demand_meet = optimization.compute_metrics(
            solar_size, wind_size, battery_size, df, config_list,colocation_option=colocation_choice
        )

        # Cost Calculation for the optimized sizes
        output_path = f"/Users/harikrishnan/Desktop/KA_validation/Karnataka_Simulation_Results/simulation_results_prv11{config_choice}.xlsx"
        df_sim, cost_dict, max_grid_cost_val = simulation.simulate_battery(df.copy(), solar_size, wind_size, battery_size, colocation_choice, RE_util, output_path, user_demand)

        # Debugging: Print demand charge details
        total_demand_charge = cost_dict.get('total_demand_charge', 0)
        total_energy_charge_full_grid = cost_dict.get('total_energy_charge_full_grid', 0)
        total_demand_charge_full_grid = cost_dict.get('total_demand_charge_full_grid', 0)
        total_demand = df_sim['Data center Actual Demand'].sum()
        unit_grid_cost = ((total_energy_charge_full_grid + total_demand_charge_full_grid) / (total_demand * 1000))*10**7

        print(f'REutilization% for  {config_choice}:{RE_util}')
        print(f'Battery_cost for {config_choice}:{cost_dict.get('battery_cost_per_MWh', 0)}')
        print(f'Peak Demand Met for {config_choice}:{peak_met}')
        print(f'Annual Demand Met for {config_choice}:{annual_dc_demand_meet}')

        # Compute total costs from the returned dictionary
        total_cost_without_exports = (
            cost_dict.get('total_solar_cost', 0) +
            cost_dict.get('total_wind_cost', 0) +
            cost_dict.get('total_battery_cost', 0) +
            cost_dict.get('total_grid_cost', 0) +
            cost_dict.get('total_market_cost', 0)
        )
        total_export_revenue = cost_dict.get('total_export_revenue', 0)
        total_cost = total_cost_without_exports - total_export_revenue

        total_cost_with_demand_charge_with_selling = total_cost + total_demand_charge
        total_cost_with_demand_charge_without_selling = total_cost_without_exports + total_demand_charge

        total_cost_per_kWh = (total_cost_with_demand_charge_with_selling / (total_demand * 1000)) * 10**7
        cost_without_selling = (total_cost_with_demand_charge_without_selling / (total_demand * 1000)) * 10**7
        cost_per_kWh_with_demand = (total_cost_with_demand_charge_with_selling / (total_demand * 1000)) * 10**7
        cost_per_kWh_with_demand_without_selling = (total_cost_with_demand_charge_without_selling / (total_demand * 1000)) * 10**7

        # Calculation of metrics
        total_battery_discharge = df_sim['Battery Discharge (MW)'].sum()
        total_battery_charge = df_sim['Battery Charge (MW)'].sum()
        total_grid_energy = df_sim['Grid Energy (MW)'].sum()
        total_market_energy = df_sim['Market Energy (MW)'].sum()

        demand_meet_by_storage = (total_battery_discharge / total_demand) * 100
        deficit_meet_from_grid = (total_grid_energy / total_demand) * 100
        deficit_meet_from_green_market = (total_market_energy / total_demand) * 100

        total_generation = df_sim['Total Generation (MW)'].sum()
        total_excess_exported = df_sim['Excess Exported (MW)'].sum()
        demand_meet_from_re = ((total_generation - total_excess_exported - total_battery_charge) / total_demand) * 100

        # results
        results.append({
            'Energy Mix Combination': config_name,
            'Solar (MW)': solar_size if 'Solar' in config_list else 0,
            'Wind (MW)': wind_size if 'Wind' in config_list else 0,
            'Storage Capacity (MW)': battery_size / config.BATTERY_BACKUP_HOURS if 'Battery' in config_list else 0,
            'Storage Capacity (MWh)': battery_size if 'Battery' in config_list else 0,
            'RE Utilization (%)': RE_util,
            'BESS Utilization (%)': batt_util,
            'Peak Demand Met (%)': peak_met,
            'Demand Meet from RE (%)': demand_meet_from_re,
            'Demand Meet by Storage (%)': demand_meet_by_storage,
            'Deficit Meet from Grid (%)': deficit_meet_from_grid,
            'Deficit Meet from Green Market (%)': deficit_meet_from_green_market,
            'Annual Solar Cost (Cr)': cost_dict.get('total_solar_cost', 0),
            'Annual Wind Cost (Cr)': cost_dict.get('total_wind_cost', 0),
            'Annual BESS Cost (Cr)': cost_dict.get('total_battery_cost', 0),
            'Annual Open Market Cost (Cr)': cost_dict.get('total_market_cost', 0),
            'Annual Grid Cost (Cr)': cost_dict.get('total_grid_cost', 0),
            'Annual Operational Cost (Cr)': total_cost_without_exports + total_demand_charge,
            'LCOE (Without Selling Excess)': cost_without_selling,
            'Cost Recovery by Selling RE Energy to Open Market': total_export_revenue,
            'Total Annual Operational Cost (Cr)': total_cost_with_demand_charge_with_selling,
            'LCOE (With Selling Excess)': total_cost_per_kWh,
            'Grid Annual Cost-100%_import (Cr)': total_energy_charge_full_grid + total_demand_charge_full_grid,
            'Grid Avg Cost Per Unit': unit_grid_cost
        })

    # Convert results to DataFrame and save to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel("/Users/harikrishnan/Desktop/KA_validation/karnataka_results_prv11.xlsx", index=False)

    end_time = time.time()
    print("\nExecution time:", end_time - start_time)

if __name__ == "__main__":
    main()
