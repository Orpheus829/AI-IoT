"""
Configuration Module for AIoT Work System Design
Contains all system parameters, constants, and configuration settings
Based on the mathematical framework from Chapters 4-13
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class SystemParameters:
    """Core system parameters from the mathematical framework"""
    
    # Stochastic Parameters (Chapter 8)
    TASK_ARRIVAL_RATE: float = 40.0  # λ - tasks/hour (Poisson)
    BASE_SERVICE_RATE: float = 60.0  # μ_base - tasks/hour
    
    # Semantic Parameters (Chapter 4)
    COGNITIVE_RESISTANCE_COEFF: float = 1.2  # κ - sensitivity to ambiguity
    CRITICAL_AMBIGUITY_THRESHOLD: float = 0.35  # ψ_crit - safety threshold
    SAFETY_CONFIDENCE_THRESHOLD: float = 0.85  # δ - minimum confidence
    
    # Network Parameters (Chapter 5)
    SAMPLING_FREQ_MULTIPLIER: int = 2  # Nyquist-Shannon compliance
    MQTT_QOS_LEVEL: int = 1
    TARGET_LATENCY_MS: float = 100.0  # Maximum allowed latency
    
    # Cognitive Parameters (Chapter 9)
    HICK_HYMAN_A: float = 0.15  # Reaction time constant (seconds)
    HICK_HYMAN_B: float = 0.25  # Log choice scaling factor
    FATIGUE_ACCUMULATION_RATE: float = 0.05  # k1 - fatigue rate
    RECOVERY_RATE: float = 0.08  # k2 - recovery rate
    MAX_COGNITIVE_LOAD: float = 100.0  # Maximum workload score
    OPTIMAL_STRESS_LEVEL: float = 0.5  # S0 for Yerkes-Dodson
    STRESS_VARIANCE: float = 0.1  # σ for Yerkes-Dodson
    
    # Reliability Parameters (Chapter 6)
    WEIBULL_SHAPE_INFANT: float = 0.8  # α < 1: infant mortality
    WEIBULL_SHAPE_RANDOM: float = 1.0  # α = 1: random failure
    WEIBULL_SHAPE_WEAROUT: float = 2.5  # α > 1: wear-out
    WEIBULL_SCALE: float = 1000.0  # β - characteristic life (hours)
    
    # Reinforcement Learning Parameters (Chapter 11)
    RL_LEARNING_RATE: float = 0.001  # η - Q-learning rate
    RL_DISCOUNT_FACTOR: float = 0.95  # γ - future reward discount
    RL_EXPLORATION_RATE: float = 0.1  # ε - exploration probability
    RL_EPISODES: int = 1000
    
    # Kalman Filter Parameters (Chapter 5)
    PROCESS_NOISE_COV: float = 0.01  # Q - system uncertainty
    MEASUREMENT_NOISE_COV: float = 0.1  # R - sensor noise
    
    # Simulation Parameters (Chapter 13)
    SIMULATION_DURATION_HOURS: int = 8  # Work shift duration
    SIMULATION_ITERATIONS: int = 10000  # DES sample size
    TIME_STEP: float = 0.1  # Simulation granularity (seconds)
    
    # WSD Parameters (Chapter 10)
    CONTEXT_WINDOW_SIZE: int = 5  # Words before/after for disambiguation
    WSD_CONFIDENCE_THRESHOLD: float = 0.7
    MAX_SENSE_CANDIDATES: int = 5
    
    # Ergonomic Parameters (Chapter 7)
    POSTURE_RISK_WEIGHTS: Dict[str, float] = None
    ENVIRONMENT_RISK_THRESHOLDS: Dict[str, Tuple[float, float]] = None
    
    def __post_init__(self):
        """Initialize complex parameters after dataclass creation"""
        if self.POSTURE_RISK_WEIGHTS is None:
            self.POSTURE_RISK_WEIGHTS = {
                'back_angle': 0.35,
                'wrist_rotation': 0.25,
                'shoulder_elevation': 0.20,
                'neck_angle': 0.20
            }
        
        if self.ENVIRONMENT_RISK_THRESHOLDS is None:
            self.ENVIRONMENT_RISK_THRESHOLDS = {
                'temperature': (18.0, 26.0),  # Celsius
                'humidity': (30.0, 70.0),      # Percentage
                'noise': (0.0, 85.0),          # dB
                'light': (300.0, 750.0)        # Lux
            }
    
    def calculate_critical_ambiguity(self) -> float:
        """
        Calculate ψ_crit from Equation 12.4
        ψ_crit = (1/κ) * ln(μ_base / λ)
        """
        return (1.0 / self.COGNITIVE_RESISTANCE_COEFF) * \
               np.log(self.BASE_SERVICE_RATE / self.TASK_ARRIVAL_RATE)
    
    def calculate_phase_margin_constraint(self, crossover_freq: float) -> float:
        """
        Calculate maximum latency for stability
        L_total < φ_m / ω_c
        """
        phase_margin = 60.0  # degrees (typical for stable systems)
        return np.deg2rad(phase_margin) / crossover_freq


# Constants for Semantic Disambiguation
WSD_SENSE_INVENTORY = {
    'line': [
        'assembly_line',
        'electrical_line', 
        'hydraulic_line',
        'communication_line',
        'queue_line'
    ],
    'arm': [
        'robot_arm',
        'human_arm',
        'mechanical_arm'
    ],
    'check': [
        'inspect_visually',
        'verify_electronically',
        'test_functionally',
        'query_status'
    ],
    'resistance': [
        'electrical_resistance',
        'mechanical_resistance',
        'psychological_resistance'
    ],
    'load': [
        'mechanical_load',
        'electrical_load',
        'cognitive_load',
        'workload'
    ],
    'stop': [
        'emergency_stop',
        'normal_stop',
        'pause',
        'halt_temporarily'
    ]
}

# Machine States for Markov Models
MACHINE_STATES = {
    0: 'Healthy',
    1: 'Degrading', 
    2: 'Failed'
}

# Task Types for RL Agent
TASK_TYPES = {
    0: 'Routine_Inspection',
    1: 'Complex_Repair',
    2: 'Emergency_Response',
    3: 'Preventive_Maintenance'
}

# Failure Modes (from AI4I dataset)
FAILURE_MODES = {
    'TWF': 'Tool_Wear_Failure',
    'HDF': 'Heat_Dissipation_Failure',
    'PWF': 'Power_Failure',
    'OSF': 'Overstrain_Failure',
    'RNF': 'Random_Failure'
}

# Color schemes for visualization
COLOR_SCHEME = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#06A77D',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'neutral': '#6C757D'
}

# Create global instance
config = SystemParameters()

# Helper functions
def get_transition_matrix(health_state: str = 'normal') -> np.ndarray:
    """
    Generate Markov transition matrix for machine states
    Based on Chapter 8.4
    """
    if health_state == 'normal':
        return np.array([
            [0.95, 0.04, 0.01],  # Healthy -> [H, D, F]
            [0.20, 0.70, 0.10],  # Degrading -> [H, D, F]
            [0.00, 0.00, 1.00]   # Failed -> [H, D, F]
        ])
    elif health_state == 'stressed':
        return np.array([
            [0.85, 0.12, 0.03],
            [0.10, 0.60, 0.30],
            [0.00, 0.00, 1.00]
        ])
    else:
        raise ValueError(f"Unknown health state: {health_state}")

def semantic_efficiency(ambiguity: float, kappa: float = None) -> float:
    """
    Calculate η(ψ) = e^(-κψ) from Equation 12.1
    
    Args:
        ambiguity: Semantic ambiguity score (0-1)
        kappa: Cognitive resistance coefficient (default from config)
    
    Returns:
        Efficiency multiplier (0-1)
    """
    if kappa is None:
        kappa = config.COGNITIVE_RESISTANCE_COEFF
    return np.exp(-kappa * ambiguity)

def effective_service_rate(ambiguity: float, base_rate: float = None) -> float:
    """
    Calculate μ_eff = μ_base * e^(-κψ) from Equation 12.2
    
    Args:
        ambiguity: Semantic ambiguity score
        base_rate: Base service rate (default from config)
    
    Returns:
        Effective service rate
    """
    if base_rate is None:
        base_rate = config.BASE_SERVICE_RATE
    return base_rate * semantic_efficiency(ambiguity)

def system_latency(ambiguity: float, arrival_rate: float = None, 
                   base_rate: float = None) -> float:
    """
    Calculate W_q(ψ) from Equation 12.3
    
    Returns:
        Average waiting time (infinity if unstable)
    """
    if arrival_rate is None:
        arrival_rate = config.TASK_ARRIVAL_RATE
    if base_rate is None:
        base_rate = config.BASE_SERVICE_RATE
    
    mu_eff = effective_service_rate(ambiguity, base_rate)
    
    if mu_eff <= arrival_rate:
        return np.inf  # System unstable
    
    return arrival_rate / (mu_eff * (mu_eff - arrival_rate))

if __name__ == "__main__":
    # Test configuration
    print("System Configuration Test")
    print("=" * 50)
    print(f"Critical Ambiguity Threshold: {config.calculate_critical_ambiguity():.4f}")
    print(f"Task Arrival Rate: {config.TASK_ARRIVAL_RATE} tasks/hr")
    print(f"Base Service Rate: {config.BASE_SERVICE_RATE} tasks/hr")
    print(f"\nTesting Semantic Functions:")
    print(f"η(0.0) = {semantic_efficiency(0.0):.4f}")
    print(f"η(0.3) = {semantic_efficiency(0.3):.4f}")
    print(f"W_q(0.0) = {system_latency(0.0):.2f} seconds")
    print(f"W_q(0.3) = {system_latency(0.3):.2f} seconds")
