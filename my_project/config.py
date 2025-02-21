# Global configuration parameters
BASELINE_DEMAND = 100.0  # MW

# PPA Tariffs (Rs/MWh)
SOLAR_PPA = 3000
WIND_PPA = 3500

# For Demand Charge
POWER_FACTOR = 0.9
CONTRACT_DEMAND = 100
DEMAND_CHARGE_PER_MVA =340000   # Rs for Karnataka
#DEMAND_CHARGE_PER_MVA =475000    # Rs for Gujarat


# Grid cost during peak and off peak periods
# GRID_COST_PEAK = 6500
# GRID_COST_OFFPEAK = 5650

# Transmission cost when not colocated (Rs/MWh)
TRANSMISSION_COST = 1850     # For Karnataka
#TRANSMISSION_COST = 2500      # For Gujarat

# Battery cost parameters
BATTERY_FIXED_COST_FACTOR = 4200000  # Annual fixed cost factor
BATTERY_BACKUP_HOURS = 4  # For backup duration

INITIAL_SOC= 0.4

# Battery charging and discharging efficiency
BATTERY_CHARGING_EFFICIENCY = 0.95
BATTERY_DISCHARGING_EFFICIENCY = 0.95
RTE = BATTERY_CHARGING_EFFICIENCY*BATTERY_DISCHARGING_EFFICIENCY    # Round trip efficiency

# Other cost parameters for fallback (cost while selling)
FALLBACK_SOLAR_COST = 3000
FALLBACK_WIND_COST = 3500

# Optimization parametersx
OPT_MAXITER = 250
OPT_POPSIZE = 10
OPT_TOL = 1e-8
