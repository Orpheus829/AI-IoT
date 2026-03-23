"""
Dataset Loader Module for AIoT Work System Design
Handles loading and preprocessing of AI4I 2020 Predictive Maintenance Dataset
Reference: UCI ML Repository / Kaggle (shivamb/machine-predictive-maintenance-classification)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class AI4IDataLoader:
    """
    Loads and preprocesses the AI4I 2020 Predictive Maintenance Dataset
    
    Dataset contains:
    - 10,000 data points
    - Sensor readings: air temp, process temp, rotational speed, torque, tool wear
    - Failure labels: TWF, HDF, PWF, OSF, RNF
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data loader
        
        Args:
            data_path: Path to ai4i2020.csv (optional)
        """
        self.data_path = data_path
        self.df_raw = None
        self.df_processed = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load the AI4I dataset from CSV
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Raw dataframe
        """
        if filepath is None:
            filepath = self.data_path
        
        if filepath is None:
            raise ValueError("No data path provided")
        
        self.df_raw = pd.read_csv(filepath)
        print(f"Loaded {len(self.df_raw)} records from {filepath}")
        print(f"Columns: {list(self.df_raw.columns)}")
        
        return self.df_raw
    
    def preprocess_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Preprocess the dataset for modeling
        
        Steps:
        1. Handle missing values
        2. Extract features
        3. Create derived variables
        4. Normalize sensors
        
        Args:
            df: Input dataframe (uses self.df_raw if None)
            
        Returns:
            Processed dataframe
        """
        if df is None:
            df = self.df_raw.copy()
        else:
            df = df.copy()
        
        # Expected columns (may vary slightly by source)
        # Standard columns: Type, Air temperature [K], Process temperature [K],
        #                   Rotational speed [rpm], Torque [Nm], Tool wear [min],
        #                   Machine failure, TWF, HDF, PWF, OSF, RNF
        
        # Rename columns for easier access
        column_mapping = {
            'Air temperature [K]': 'air_temp',
            'Process temperature [K]': 'process_temp',
            'Rotational speed [rpm]': 'rotational_speed',
            'Torque [Nm]': 'torque',
            'Tool wear [min]': 'tool_wear',
            'Machine failure': 'machine_failure'
        }
        
        # Apply renaming only for columns that exist
        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)
        
        # Convert temperatures from Kelvin to Celsius if needed
        if 'air_temp' in df.columns and df['air_temp'].mean() > 100:
            df['air_temp'] = df['air_temp'] - 273.15
            df['process_temp'] = df['process_temp'] - 273.15
        
        # Create derived features
        if 'process_temp' in df.columns and 'air_temp' in df.columns:
            df['temp_difference'] = df['process_temp'] - df['air_temp']
        
        if 'rotational_speed' in df.columns and 'torque' in df.columns:
            df['power'] = df['rotational_speed'] * df['torque'] / 9550  # kW
        
        if 'tool_wear' in df.columns:
            df['tool_wear_rate'] = df['tool_wear'] / (df.index + 1)  # Cumulative rate
        
        # Create health state labels (for Markov model)
        if 'machine_failure' in df.columns and 'tool_wear' in df.columns:
            df['health_state'] = 'Healthy'
            df.loc[(df['tool_wear'] > 150) & (df['machine_failure'] == 0), 'health_state'] = 'Degrading'
            df.loc[df['machine_failure'] == 1, 'health_state'] = 'Failed'
        
        # Create failure mode column (one-hot encoded failures)
        failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        existing_failure_cols = [c for c in failure_cols if c in df.columns]
        
        if existing_failure_cols:
            df['failure_mode'] = 'None'
            for col in existing_failure_cols:
                df.loc[df[col] == 1, 'failure_mode'] = col
        
        # Encode categorical variables
        if 'Type' in df.columns:
            if 'Type' not in self.label_encoders:
                self.label_encoders['Type'] = LabelEncoder()
                df['type_encoded'] = self.label_encoders['Type'].fit_transform(df['Type'])
            else:
                df['type_encoded'] = self.label_encoders['Type'].transform(df['Type'])
        
        if 'health_state' in df.columns:
            state_mapping = {'Healthy': 0, 'Degrading': 1, 'Failed': 2}
            df['health_state_encoded'] = df['health_state'].map(state_mapping)
        
        self.df_processed = df
        return df
    
    def normalize_features(self, features: list, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Normalize specified features using StandardScaler
        
        Args:
            features: List of feature column names
            df: Dataframe to normalize (uses self.df_processed if None)
            
        Returns:
            Dataframe with normalized features
        """
        if df is None:
            df = self.df_processed.copy()
        else:
            df = df.copy()
        
        existing_features = [f for f in features if f in df.columns]
        
        if existing_features:
            df[existing_features] = self.scaler.fit_transform(df[existing_features])
        
        return df
    
    def add_semantic_ambiguity(self, df: Optional[pd.DataFrame] = None, 
                               seed: int = 42) -> pd.DataFrame:
        """
        Add simulated semantic ambiguity scores to the dataset
        This represents the clarity of operator instructions in AIoT environment
        
        Based on Chapter 4: ambiguity increases with:
        - High cognitive load (complex tasks)
        - Multiple failure modes active
        - Degraded machine state
        
        Args:
            df: Dataframe to augment
            seed: Random seed for reproducibility
            
        Returns:
            Dataframe with ambiguity column
        """
        if df is None:
            df = self.df_processed.copy()
        else:
            df = df.copy()
        
        np.random.seed(seed)
        
        # Base ambiguity: small random noise
        base_ambiguity = np.random.uniform(0.05, 0.15, len(df))
        
        # Increase ambiguity for complex scenarios
        ambiguity_multiplier = np.ones(len(df))
        
        if 'health_state_encoded' in df.columns:
            # Degrading/Failed states increase ambiguity
            ambiguity_multiplier += df['health_state_encoded'] * 0.15
        
        if 'failure_mode' in df.columns:
            # Multiple concurrent issues increase ambiguity
            ambiguity_multiplier[df['failure_mode'] != 'None'] += 0.2
        
        if 'tool_wear' in df.columns:
            # High tool wear adds uncertainty
            normalized_wear = df['tool_wear'] / df['tool_wear'].max()
            ambiguity_multiplier += normalized_wear * 0.1
        
        df['semantic_ambiguity'] = np.clip(base_ambiguity * ambiguity_multiplier, 0.0, 1.0)
        
        return df
    
    def add_cognitive_load(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Add simulated cognitive load based on system state
        
        Cognitive load increases with:
        - Number of active alerts
        - System complexity
        - Time pressure (simulated)
        
        Args:
            df: Dataframe to augment
            
        Returns:
            Dataframe with cognitive_load column
        """
        if df is None:
            df = self.df_processed.copy()
        else:
            df = df.copy()
        
        # Base cognitive load
        base_load = 30.0  # Nominal load
        
        cognitive_load = np.full(len(df), base_load)
        
        if 'machine_failure' in df.columns:
            # Failures increase cognitive demand
            cognitive_load += df['machine_failure'] * 25.0
        
        if 'health_state_encoded' in df.columns:
            # Degrading state adds monitoring burden
            cognitive_load += df['health_state_encoded'] * 10.0
        
        # Add temporal variation (fatigue over shift)
        time_factor = np.linspace(0, 1, len(df))
        cognitive_load += time_factor * 15.0  # Fatigue accumulation
        
        # Add random daily variation
        cognitive_load += np.random.normal(0, 5, len(df))
        
        df['cognitive_load'] = np.clip(cognitive_load, 0, 100)
        
        return df
    
    def create_time_series(self, df: Optional[pd.DataFrame] = None, 
                          freq: str = '1H') -> pd.DataFrame:
        """
        Convert dataset to time-series format with datetime index
        
        Args:
            df: Dataframe to convert
            freq: Frequency string (e.g., '1H' for hourly)
            
        Returns:
            Time-indexed dataframe
        """
        if df is None:
            df = self.df_processed.copy()
        else:
            df = df.copy()
        
        # Create datetime index
        start_date = pd.Timestamp('2020-01-01 08:00:00')
        df['timestamp'] = pd.date_range(start=start_date, periods=len(df), freq=freq)
        df = df.set_index('timestamp')
        
        return df
    
    def split_data(self, df: Optional[pd.DataFrame] = None, 
                   test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets
        
        Args:
            df: Dataframe to split
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            (train_df, test_df)
        """
        if df is None:
            df = self.df_processed
        
        train_df, test_df = train_test_split(df, test_size=test_size, 
                                             random_state=random_state)
        
        return train_df, test_df
    
    def get_summary_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Calculate summary statistics for the dataset
        
        Returns:
            Dictionary of statistics
        """
        if df is None:
            df = self.df_processed
        
        stats = {
            'total_records': len(df),
            'failure_rate': df['machine_failure'].mean() if 'machine_failure' in df.columns else None,
            'avg_ambiguity': df['semantic_ambiguity'].mean() if 'semantic_ambiguity' in df.columns else None,
            'avg_cognitive_load': df['cognitive_load'].mean() if 'cognitive_load' in df.columns else None,
        }
        
        if 'health_state' in df.columns:
            stats['health_distribution'] = df['health_state'].value_counts().to_dict()
        
        if 'failure_mode' in df.columns:
            stats['failure_modes'] = df[df['failure_mode'] != 'None']['failure_mode'].value_counts().to_dict()
        
        return stats


def create_sample_dataset(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic sample dataset for testing when real data unavailable
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed
        
    Returns:
        Synthetic dataframe with same structure as AI4I dataset
    """
    np.random.seed(seed)
    
    df = pd.DataFrame({
        'Type': np.random.choice(['L', 'M', 'H'], n_samples),
        'air_temp': np.random.normal(25, 2, n_samples),  # Celsius
        'process_temp': np.random.normal(35, 5, n_samples),
        'rotational_speed': np.random.normal(1500, 200, n_samples),
        'torque': np.random.normal(40, 10, n_samples),
        'tool_wear': np.random.randint(0, 250, n_samples),
    })
    
    # Calculate derived features first
    df['temp_difference'] = df['process_temp'] - df['air_temp']
    df['power'] = df['rotational_speed'] * df['torque'] / 9550
    
    # Generate failures based on conditions
    df['machine_failure'] = 0
    
    # Tool wear failure
    df['TWF'] = ((df['tool_wear'] > 200) & (np.random.rand(n_samples) < 0.6)).astype(int)
    
    # Heat dissipation failure
    df['HDF'] = ((df['temp_difference'] > 12) & (np.random.rand(n_samples) < 0.5)).astype(int)
    
    # Power failure
    df['PWF'] = ((df['power'] > 8) & (np.random.rand(n_samples) < 0.4)).astype(int)
    
    # Overstrain failure
    df['OSF'] = ((df['torque'] > 60) & (np.random.rand(n_samples) < 0.3)).astype(int)
    
    # Random failures
    df['RNF'] = (np.random.rand(n_samples) < 0.01).astype(int)
    
    # Overall failure
    df['machine_failure'] = (df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum(axis=1) > 0).astype(int)
    
    return df


if __name__ == "__main__":
    # Test the loader
    print("Testing AI4I Data Loader")
    print("=" * 50)
    
    # Create sample data
    df_sample = create_sample_dataset(1000)
    print(f"\nCreated sample dataset: {df_sample.shape}")
    
    # Initialize loader
    loader = AI4IDataLoader()
    loader.df_raw = df_sample
    
    # Preprocess
    df_processed = loader.preprocess_data()
    print(f"Processed dataset: {df_processed.shape}")
    print(f"Columns: {list(df_processed.columns)}")
    
    # Add AIoT-specific features
    df_aiot = loader.add_semantic_ambiguity(df_processed)
    df_aiot = loader.add_cognitive_load(df_aiot)
    
    # Summary
    stats = loader.get_summary_statistics(df_aiot)
    print(f"\nSummary Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
