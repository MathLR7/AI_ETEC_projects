import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report


def get_data(file1 = "Dataset.xlsx", file2="CycleOfLife.xlsx"): 

    """ returns entire dataset raw data and cycle of life data """

    Dataset = pd.read_excel(file1)
    CycleOfLife = pd.read_excel(file2)

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

    labeled_df = pd.concat([top_45, bottom_45]).sort_values("cell_id")
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
        
        # 1. Total time accumulated above 3.9 V
        # Garantimos que o delta de tempo não seja negativo caso o step_time resete
        group['time_delta_s'] = group.groupby('cycle')['step_time_s'].diff().fillna(0)
        
        # For security, we'll assume we can approximate row duration via time delta or step duration.
        
        features['time_above_3_9V_total'] = group.loc[group['voltage_V'] > 3.9, 'time_delta_s'].sum()
        
        # 2. First cycle data subset
        cycle_1 = group[group['cycle'] == 1]
        
        if not cycle_1.empty:
            cycle_1_time_delta = cycle_1.loc[:, 'step_time_s'].diff().fillna(0)
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
            # Usamos iloc[0] para pegar o primeiro evento de relaxação
            event = relaxation_starts.iloc[0]
            start_idx = group.index.get_loc(relaxation_starts.index[0])
            
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


def test_cell(df: pd.DataFrame, cell: int = 1, cycle: int = 0, plot: bool = True):
    print(f'possible cycles to test: {df.loc[(df["cell_id"]==cell)]["cycle"].unique()}')

    test_df = df.loc[(df["cell_id"]==cell) & (df["cycle"]==cycle)].copy()
    test_df["time_s"] = test_df["time_s"] - test_df["time_s"].iloc[0]
    test_df["time_h"] = test_df["time_s"] / 3600
    test_df["current_mA"] = test_df["current_A"] * 1000


    if plot:

        ax = test_df.plot(
            x = "time_h",
            y = "voltage_V",
            color="orange",
            label = "voltage",
            figsize = (10, 5)

        )

        test_df.plot(
            x = "time_h",
            y = "current_mA",
            secondary_y = True,
            color = "blue",
            label = "current", 
            ax = ax
        )

        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Voltage [V]")
        ax.right_ax.set_ylabel("Current [mA]")

        ax.set_title(f'Voltage and Current profile for cell: {cell}, cycle: {cycle}')


    return test_df

def filtered_features(df_features: pd.DataFrame, df_cycle: pd.DataFrame) -> pd.DataFrame:
   # 1. Standardize df_cycle to have cell_id as index for easy mapping
    if 'cell_id' in df_cycle.columns:
        cycle_mapped = df_cycle.set_index('cell_id')
    else:
        cycle_mapped = df_cycle

    # 2. Extract valid cell IDs from the cycle data
    cycle_ids = cycle_mapped.index.unique()
        
    # 3. Filter df_features rows to keep only matching cell IDs
    if 'cell_id' in df_features.columns:
        filtered = df_features[df_features['cell_id'].isin(cycle_ids)].copy()
        # Add the label column using map
        filtered['label'] = filtered['cell_id'].map(cycle_mapped['label'])
    else:
        # Fallback if cell_id is the index of df_features
        filtered = df_features[df_features.index.isin(cycle_ids)].copy()
        # Add the label column using map on the index
        filtered['label'] = filtered.index.map(cycle_mapped['label'])

    return filtered

from sklearn.model_selection import train_test_split


def separeData(df: pd.DataFrame, target_column: str = 'label', test_size: float = 0.2, random_state: int = 42):
    columns_to_drop = [target_column]
    if 'cell_id' in df.columns:
        columns_to_drop.append('cell_id')
    X = df.drop(columns=columns_to_drop)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def create_pca(filtered_features, random_seed:int = 42):

    """
    From an already filtered features df, lower dimension using PCA and returns 
    the new x and y arrays for the model training and validation
    """

    X_train, X_test, y_train, y_test = separeData(filtered_features, random_state=random_seed)

    scaler = sklearn.preprocessing.StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    pca = sklearn.decomposition.PCA(n_components=0.95)
    X_train_pca_array = pca.fit_transform(X_train_scaled)

    pca_columns = [f"PC{i+1}" for i in range(X_train_pca_array.shape[1])]

    X_train_pca = pd.DataFrame(
        X_train_pca_array,
        columns=pca_columns,
        index=X_train.index
    )

    X_test_scaled = scaler.transform(X_test)
    X_test_pca_array = pca.transform(X_test_scaled)

    X_test_pca = pd.DataFrame(
        X_test_pca_array,
        columns=pca_columns,
        index=X_test.index
    )

    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 5))
    plt.plot(
        np.arange(1, len(explained_variance) + 1),
        np.cumsum(explained_variance),
        marker="o"
    )

    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.grid(True)
    plt.show()

    return X_train_pca, X_test_pca, y_train, y_test



def supervised_learning(X_train_pca, X_test_pca, y_train, y_test, n_neighbors=5, plot=True, feature1='PC1', feature2='PC2'):

    models = {
        
    # "Logistic Regression": LogisticRegression(max_iter=1000),
     "KNN": KNeighborsClassifier(n_neighbors),
    # "SVM Linear": SVC(kernel="linear"),
    # "SVM RBF": SVC(kernel="rbf")

    }

    all_results = {}

    for name, model in models.items():
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)

        results = pd.DataFrame({
            "actual_label": y_test,
            "predicted_label": y_pred
        }, index=X_test_pca.index)

        results["correct_prediction"] = (
            results["actual_label"].astype(str) == results["predicted_label"].astype(str)
        )

        all_results[name] = results

        print(name)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print("-" * 50)


    if plot:
        # Convert labels to arrays/series with matching indexes
        y_test_pred = pd.Series(y_pred, index=X_test_pca.index, name="predicted_label")
        y_test_true = pd.Series(y_test, index=X_test_pca.index, name="true_label")

        labels = np.unique(np.concatenate([y_train.astype(str), y_test_true.astype(str), y_test_pred.astype(str)]))

        colors = {
            labels[0]: "#CC0000",
            labels[1]: "#0000CC"
        }

        plt.figure(figsize=(8, 6))

        # Plot training data using true labels
        for label in labels:
            train_mask = y_train.astype(str) == label

            plt.scatter(
                X_train_pca.loc[train_mask, feature1],
                X_train_pca.loc[train_mask, feature2],
                color=colors[label],
                edgecolor="black",
                s=70,
                alpha=0.65,
                label=f"Train true: {label}"
            )

        # Plot test data using predicted labels
        for label in labels:
            pred_mask = y_test_pred.astype(str) == label

            plt.scatter(
                X_test_pca.loc[pred_mask, feature1],
                X_test_pca.loc[pred_mask, feature2],
                color=colors[label],
                marker="X",
                edgecolor="black",
                s=130,
                label=f"Test predicted: {label}"
            )

        # Highlight wrong predictions
        wrong_mask = y_test_true.astype(str) != y_test_pred.astype(str)

        plt.scatter(
            X_test_pca.loc[wrong_mask, feature1],
            X_test_pca.loc[wrong_mask, feature2],
            facecolors="none",
            edgecolors="gold",
            linewidths=2.5,
            s=220,
            label="Incorrect prediction"
        )

        plt.xlabel(f"feature {feature1}")
        plt.ylabel(f"feature {feature2}")
        plt.title("KNN Predictions on PCA Features")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig("knn_predictions_pca.png", dpi=300, bbox_inches="tight")
        plt.show()
        
    return all_results


