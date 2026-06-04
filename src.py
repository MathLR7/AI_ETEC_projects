import pandas as pd

def get_data(file1 = "Dataset.xlsx", file2="CycleOfLife.xlsx"): 

    """ returns entire dataset raw data and cycle of life data """

    Dataset = pd.read_excel("Dataset.xlsx")
    CycleOfLife = pd.read_excel("CycleOfLife.xlsx")

    return Dataset, CycleOfLife 

def get_quantile(df: pd.DataFrame) -> pd.DataFrame: 

    """
    Gets Cycle of life data for each cell and label them as "good" or "bad"
    good = top 45% cycle of life
    bad = bottom 45% cycle of life
    middle 10% gets excluded
    """

    bottom_cutoff = df["Cycle of Life"].quantile(0.45)
    top_cutoff = df["Cycle of Life"].quantile(0.55)

    bottom_45 = df[df["Cycle of Life"] <= bottom_cutoff].copy()
    bottom_45["label"] = "bad"

    top_45 = df[df["Cycle of Life"] >= top_cutoff].copy()
    top_45["label"] = "good"

    labeled_df = pd.concat([top_45, bottom_45]).sort_values("Cycle of Life")
    return labeled_df

import pandas as pd
import numpy as np

def extract_battery_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts Classes A, B, and C features from raw formation/early-cycle battery data.
    
    Assumes df contains columns:
    - 'cell_id': Unique identifier for each battery cell
    - 'voltage_V': Voltage measurements
    - 'step_time_s': Time elapsed within the current step/cycle (or overall time step delta)
    - 'current_A': Current measurements
    - 'Q_charge_Ah': Charge capacity
    - 'E_charge_Wh': Charge energy
    - 'Q_discharge_Ah': Discharge capacity
    - 'cycle': Integer indicating the cycle index (assumed 1 for first cycle)
    
    Returns:
    - pd.DataFrame: A dataframe containing one row per cell_id with all calculated features.
    """
    
    feature_list = []
    
    # Group by cell_id to process each battery cell independently
    for cell_id, group in df.groupby('cell_id'):
        features = {'cell_id': cell_id}
        
        # Ensure data is sorted sequentially
        group = group.sort_values(by=['cycle', 'step_time_s']).reset_index(drop=True)
        
        # --- CLASSE A: Statistical and Time Features (Raw Data) ---
        
        # 1. Total time accumulated above 3.9 V (using a time delta calculation per row)
        # If 'step_time_s' is cumulative time, we use diff(). If it's the duration of the step, use it directly.
        # Assuming we need to calculate the time difference between consecutive logged points:
        group['time_delta_s'] = group['step_time_s'].diff().fillna(0)
        # Alternative if step_time_s resets every step but rows are continuous: 
        # For security, we'll assume we can approximate row duration via time delta or step duration.
        
        features['time_above_3_9V_total'] = group.loc[group['voltage_V'] > 3.9, 'time_delta_s'].sum()
        
        # 2. First cycle data subset
        cycle_1 = group[group['cycle'] == 1]
        
        if not cycle_1.empty:
            cycle_1_time_delta = cycle_1['step_time_s'].diff().fillna(0)
            features['time_above_3_5V_cycle1'] = cycle_1_time_delta[cycle_1['voltage_V'] > 3.5].sum()
            features['max_Q_charge_cycle1'] = cycle_1['Q_charge_Ah'].max()
            features['max_E_charge_cycle1'] = cycle_1['E_charge_Wh'].max()
        else:
            features['time_above_3_5V_cycle1'] = np.nan
            features['max_Q_charge_cycle1'] = np.nan
            features['max_E_charge_cycle1'] = np.nan
            
        # 3. Accumulated time between 3.9 V and 4.0 V during formation
        features['time_between_3_9V_4_0V'] = group.loc[
            (group['voltage_V'] >= 3.9) & (group['voltage_V'] <= 4.0), 'time_delta_s'
        ].sum()
        
        # 4. Voltage statistics during active charging (current > 0)
        active_charge = group[group['current_A'] > 0.01]  # small threshold to avoid noise
        if not active_charge.empty:
            features['mean_voltage_charge'] = active_charge['voltage_V'].mean()
            features['var_voltage_charge'] = active_charge['voltage_V'].var()
            features['mean_current_charge'] = active_charge['current_A'].mean()
        else:
            features['mean_voltage_charge'] = np.nan
            features['var_voltage_charge'] = np.nan
            features['mean_current_charge'] = np.nan
            
        # 5. Absolute extrema
        features['max_voltage_absolute'] = group['voltage_V'].max()
        features['min_voltage_absolute'] = group['voltage_V'].min()
        
        
        # --- CLASSE B: Differential Analysis (dQ/dV, dE/dV, dV/dQ) ---
        
        # To compute derivatives robustly and avoid division by zero or infinite noise, 
        # we filter for active charging data and calculate differences.
        charge_sorted = active_charge.sort_values(by='voltage_V')
        
        if len(charge_sorted) > 1:
            dV = charge_sorted['voltage_V'].diff()
            dQ = charge_sorted['Q_charge_Ah'].diff()
            dE = charge_sorted['E_charge_Wh'].diff()
            
            # Avoid division by zero
            valid_dV = dV > 0
            valid_dQ = dQ > 0
            
            # dQ/dV
            dQ_dV = dQ[valid_dV] / dV[valid_dV]
            if not dQ_dV.empty:
                features['peak_height_dQ_dV'] = dQ_dV.max()
                features['peak_location_dQ_dV'] = charge_sorted.loc[dQ_dV.idxmax(), 'voltage_V']
            else:
                features['peak_height_dQ_dV'], features['peak_location_dQ_dV'] = np.nan, np.nan
                
            # dE/dV
            dE_dV = dE[valid_dV] / dV[valid_dV]
            if not dE_dV.empty:
                features['peak_height_dE_dV'] = dE_dV.max()
                features['peak_location_dE_dV'] = charge_sorted.loc[dE_dV.idxmax(), 'voltage_V']
            else:
                features['peak_height_dE_dV'], features['peak_location_dE_dV'] = np.nan, np.nan
                
            # dV/dQ
            dV_dQ = dV[valid_dQ] / dQ[valid_dQ]
            if not dV_dQ.empty:
                features['peak_height_dV_dQ'] = dV_dQ.max()
                features['peak_location_dV_dQ'] = charge_sorted.loc[dV_dQ.idxmax(), 'voltage_V']
            else:
                features['peak_height_dV_dQ'], features['peak_location_dV_dQ'] = np.nan, np.nan
        else:
            for f in ['dQ_dV', 'dE_dV', 'dV_dQ']:
                features[f'peak_height_{f}'] = np.nan
                features[f'peak_location_{f}'] = np.nan
                
                
        # --- CLASSE C: Electrochemical Features ---
        
        # 1. Consumed Lithium (Asymmetry between first charge and discharge)
        if not cycle_1.empty:
            total_charge_c1 = cycle_1['Q_charge_Ah'].max()
            total_discharge_c1 = cycle_1['Q_discharge_Ah'].max()
            features['consumed_lithium_loss'] = total_charge_c1 - total_discharge_c1
        else:
            features['consumed_lithium_loss'] = np.nan
            
        # 2. Voltage Drop / Relaxation (1s, 10s, 60s)
        # Find rows where current drops from charging to zero (end of charge phase)
        # We look for the exact index where current becomes ~0 after being positive.
        group['current_shifted'] = group['current_A'].shift(1)
        relaxation_starts = group[(group['current_shifted'] > 0.1) & (group['current_A'].abs() <= 0.01)]
        
        if not relaxation_starts.empty:
            # Take the first occurrence (end of first charge)
            start_idx = relaxation_starts.index[0]
            v_at_cutoff = group.loc[start_idx - 1, 'voltage_V'] # Voltage right before cutoff
            
            # Look ahead in time for 1s, 10s, 60s
            relaxation_period = group.loc[start_idx:].copy()
            relaxation_period['elapsed_rest_time'] = relaxation_period['step_time_s'] - relaxation_period['step_time_s'].iloc[0]
            
            # Helper function to get voltage closest to a specific target second
            def get_relaxation_voltage(df_rest, seconds):
                if df_rest.empty: return np.nan
                idx = (df_rest['elapsed_rest_time'] - seconds).abs().idxmin()
                return df_rest.loc[idx, 'voltage_V']
            
            features['voltage_drop_relaxation_1s'] = v_at_cutoff - get_relaxation_voltage(relaxation_period, 1)
            features['voltage_drop_relaxation_10s'] = v_at_cutoff - get_relaxation_voltage(relaxation_period, 10)
            features['voltage_drop_relaxation_60s'] = v_at_cutoff - get_relaxation_voltage(relaxation_period, 60)
        else:
            features['voltage_drop_relaxation_1s'] = np.nan
            features['voltage_drop_relaxation_10s'] = np.nan
            features['voltage_drop_relaxation_60s'] = np.nan
            
        feature_list.append(features)
        
    return pd.DataFrame(feature_list)

