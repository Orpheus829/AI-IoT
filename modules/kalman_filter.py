"""
Kalman Filter Module for Sensor Fusion
Implements the Kalman filtering algorithm from Chapter 5.3
For real-time state estimation under noisy sensor conditions
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class KalmanFilterState:
    """State representation for Kalman Filter"""
    x: np.ndarray  # State estimate
    P: np.ndarray  # Error covariance matrix
    
class KalmanFilter:
    """
    Linear Kalman Filter for sensor fusion
    
    Prediction Step (Time Update):
        x̂_{t|t-1} = F * x̂_{t-1|t-1} + B * u_t
        P_{t|t-1} = F * P_{t-1|t-1} * F^T + Q
    
    Correction Step (Measurement Update):
        K_t = P_{t|t-1} * H^T * (H * P_{t|t-1} * H^T + R)^{-1}
        x̂_t = x̂_{t|t-1} + K_t * (z_t - H * x̂_{t|t-1})
        P_t = (I - K_t * H) * P_{t|t-1}
    """
    
    def __init__(self, 
                 state_dim: int,
                 measurement_dim: int,
                 F: Optional[np.ndarray] = None,
                 H: Optional[np.ndarray] = None,
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 B: Optional[np.ndarray] = None):
        """
        Initialize Kalman Filter
        
        Args:
            state_dim: Dimension of state vector
            measurement_dim: Dimension of measurement vector
            F: State transition matrix (default: identity)
            H: Measurement matrix (default: identity)
            Q: Process noise covariance (default: 0.01 * I)
            R: Measurement noise covariance (default: 0.1 * I)
            B: Control input matrix (optional)
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        
        # State transition matrix
        self.F = F if F is not None else np.eye(state_dim)
        
        # Measurement matrix
        self.H = H if H is not None else np.eye(measurement_dim, state_dim)
        
        # Process noise covariance
        self.Q = Q if Q is not None else 0.01 * np.eye(state_dim)
        
        # Measurement noise covariance
        self.R = R if R is not None else 0.1 * np.eye(measurement_dim)
        
        # Control input matrix
        self.B = B if B is not None else np.zeros((state_dim, 1))
        
        # Initialize state
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim)
        
        # Kalman gain
        self.K = None
        
    def predict(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Prediction step: Estimate next state based on system model
        
        Args:
            u: Control input vector (optional)
            
        Returns:
            Predicted state estimate
        """
        # State prediction
        if u is not None:
            self.x = self.F @ self.x + self.B @ u
        else:
            self.x = self.F @ self.x
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x
    
    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Correction step: Update estimate based on measurement
        
        Args:
            z: Measurement vector
            
        Returns:
            Updated state estimate
        """
        # Innovation (measurement residual)
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        self.K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + self.K @ y
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.state_dim) - self.K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + self.K @ self.R @ self.K.T
        
        return self.x
    
    def filter(self, z: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Combined predict-update step
        
        Args:
            z: Measurement vector
            u: Control input (optional)
            
        Returns:
            Filtered state estimate
        """
        self.predict(u)
        return self.update(z)
    
    def get_state(self) -> KalmanFilterState:
        """Get current filter state"""
        return KalmanFilterState(x=self.x.copy(), P=self.P.copy())
    
    def reset(self, x0: Optional[np.ndarray] = None, 
              P0: Optional[np.ndarray] = None):
        """
        Reset filter to initial conditions
        
        Args:
            x0: Initial state (default: zeros)
            P0: Initial covariance (default: identity)
        """
        self.x = x0 if x0 is not None else np.zeros(self.state_dim)
        self.P = P0 if P0 is not None else np.eye(self.state_dim)


class ExtendedKalmanFilter(KalmanFilter):
    """
    Extended Kalman Filter for nonlinear systems
    Uses linearization via Jacobian matrices
    """
    
    def __init__(self, state_dim: int, measurement_dim: int,
                 f_func, h_func, f_jacobian, h_jacobian,
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None):
        """
        Initialize EKF
        
        Args:
            state_dim: State dimension
            measurement_dim: Measurement dimension
            f_func: Nonlinear state transition function
            h_func: Nonlinear measurement function
            f_jacobian: Jacobian of f (state transition)
            h_jacobian: Jacobian of h (measurement)
            Q: Process noise covariance
            R: Measurement noise covariance
        """
        super().__init__(state_dim, measurement_dim, Q=Q, R=R)
        
        self.f = f_func
        self.h = h_func
        self.f_jac = f_jacobian
        self.h_jac = h_jacobian
    
    def predict(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """EKF prediction using nonlinear dynamics"""
        # Linearize at current state
        self.F = self.f_jac(self.x, u)
        
        # Nonlinear state prediction
        self.x = self.f(self.x, u)
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x
    
    def update(self, z: np.ndarray) -> np.ndarray:
        """EKF update using nonlinear measurement model"""
        # Linearize measurement at predicted state
        self.H = self.h_jac(self.x)
        
        # Innovation
        y = z - self.h(self.x)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        self.K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + self.K @ y
        
        # Covariance update
        I_KH = np.eye(self.state_dim) - self.K @ self.H
        self.P = I_KH @ self.P
        
        return self.x


class MultiSensorFusion:
    """
    Sensor fusion for multiple heterogeneous sensors
    Implements weighted fusion and Kalman-based approaches
    """
    
    def __init__(self, n_sensors: int):
        """
        Initialize multi-sensor fusion
        
        Args:
            n_sensors: Number of sensors to fuse
        """
        self.n_sensors = n_sensors
        self.weights = np.ones(n_sensors) / n_sensors  # Equal weights initially
        
    def weighted_fusion(self, measurements: np.ndarray, 
                       weights: Optional[np.ndarray] = None) -> float:
        """
        Simple weighted average fusion
        
        z = Σ w_i * x_i from Chapter 11.4
        
        Args:
            measurements: Array of sensor readings
            weights: Sensor weights (default: equal)
            
        Returns:
            Fused measurement
        """
        if weights is None:
            weights = self.weights
        
        return np.sum(weights * measurements)
    
    def optimal_weights(self, variances: np.ndarray) -> np.ndarray:
        """
        Calculate optimal fusion weights based on sensor variances
        
        Minimum variance fusion:
        w_i = (1/σ_i²) / Σ(1/σ_j²)
        
        Args:
            variances: Array of sensor variances
            
        Returns:
            Optimal weight vector
        """
        inv_var = 1.0 / variances
        weights = inv_var / np.sum(inv_var)
        return weights
    
    def update_weights(self, errors: np.ndarray, learning_rate: float = 0.1):
        """
        Adaptively update weights based on sensor errors
        
        Args:
            errors: Recent errors for each sensor
            learning_rate: Weight adaptation rate
        """
        # Inverse error weighting
        inv_errors = 1.0 / (errors + 1e-6)
        new_weights = inv_errors / np.sum(inv_errors)
        
        # Smooth update
        self.weights = (1 - learning_rate) * self.weights + learning_rate * new_weights
        self.weights /= np.sum(self.weights)  # Normalize


def create_machine_health_kf() -> KalmanFilter:
    """
    Create Kalman Filter for machine health monitoring
    
    State: [tool_wear, wear_rate]
    Measurement: [observed_wear]
    
    Returns:
        Configured KalmanFilter instance
    """
    # State dimension: 2 (wear level, wear rate)
    # Measurement dimension: 1 (observed wear)
    
    # State transition (constant velocity model)
    dt = 1.0  # Time step
    F = np.array([
        [1.0, dt],   # wear_{t+1} = wear_t + rate * dt
        [0.0, 1.0]   # rate_{t+1} = rate_t
    ])
    
    # Measurement matrix (observe only wear, not rate)
    H = np.array([[1.0, 0.0]])
    
    # Process noise (tool wear is somewhat predictable)
    Q = np.array([
        [0.01, 0.0],
        [0.0, 0.001]
    ])
    
    # Measurement noise (sensor uncertainty)
    R = np.array([[0.5]])
    
    kf = KalmanFilter(state_dim=2, measurement_dim=1, F=F, H=H, Q=Q, R=R)
    
    return kf


def create_temperature_kf() -> KalmanFilter:
    """
    Create Kalman Filter for temperature monitoring
    
    Fuses air and process temperature readings
    
    Returns:
        Configured KalmanFilter instance
    """
    # State: [true_temp, temp_rate]
    # Measurements: [air_temp, process_temp]
    
    dt = 1.0
    F = np.array([
        [1.0, dt],
        [0.0, 0.95]  # Temperature rate decays
    ])
    
    H = np.array([
        [1.0, 0.0],  # Air temp sensor
        [1.0, 0.0]   # Process temp sensor
    ])
    
    Q = np.array([
        [0.1, 0.0],
        [0.0, 0.05]
    ])
    
    R = np.array([
        [1.0, 0.0],   # Air sensor noise
        [0.0, 0.5]    # Process sensor noise (more accurate)
    ])
    
    kf = KalmanFilter(state_dim=2, measurement_dim=2, F=F, H=H, Q=Q, R=R)
    
    return kf


if __name__ == "__main__":
    print("Kalman Filter Module Test")
    print("=" * 50)
    
    # Test basic Kalman filter
    kf = create_machine_health_kf()
    
    # Simulate noisy measurements
    np.random.seed(42)
    true_wear = np.linspace(0, 200, 100)
    noisy_measurements = true_wear + np.random.normal(0, 5, 100)
    
    filtered_estimates = []
    for z in noisy_measurements:
        x_est = kf.filter(z.reshape(1))
        filtered_estimates.append(x_est[0])
    
    filtered_estimates = np.array(filtered_estimates)
    
    # Calculate improvement
    raw_error = np.mean(np.abs(noisy_measurements - true_wear))
    filtered_error = np.mean(np.abs(filtered_estimates - true_wear))
    
    print(f"\nRaw measurement error: {raw_error:.2f}")
    print(f"Filtered estimate error: {filtered_error:.2f}")
    print(f"Improvement: {(1 - filtered_error/raw_error)*100:.1f}%")
    
    # Test sensor fusion
    print(f"\nTesting Multi-Sensor Fusion:")
    fusion = MultiSensorFusion(n_sensors=3)
    measurements = np.array([25.2, 25.5, 24.8])
    variances = np.array([1.0, 0.5, 2.0])
    
    weights = fusion.optimal_weights(variances)
    fused = fusion.weighted_fusion(measurements, weights)
    
    print(f"Measurements: {measurements}")
    print(f"Optimal weights: {weights}")
    print(f"Fused value: {fused:.2f}")
