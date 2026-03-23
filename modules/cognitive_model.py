"""
Cognitive Process Mathematics Module
Implements cognitive models from Chapter 9:
- Hick-Hyman Law
- Cognitive Load Theory
- Signal Detection Theory
- Yerkes-Dodson Law
- Decision Fatigue
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class CognitiveState:
    """Represents current cognitive state of operator"""
    workload: float  # 0-100
    fatigue: float   # 0-100
    attention: float # 0-1
    stress: float    # 0-1
    
class HickHymanModel:
    """
    Hick-Hyman Law: Decision time increases logarithmically with choices
    
    T = a + b * log₂(n)
    
    Extended for ambiguity (Chapter 12.3):
    T_total = a + b * log₂(n_nominal * (1 + ψ)²)
    """
    
    def __init__(self, a: float = 0.15, b: float = 0.25):
        """
        Initialize Hick-Hyman model
        
        Args:
            a: Base reaction time (seconds)
            b: Logarithmic scaling factor
        """
        self.a = a
        self.b = b
    
    def decision_time(self, n_choices: int, ambiguity: float = 0.0) -> float:
        """
        Calculate decision time
        
        Args:
            n_choices: Number of available choices
            ambiguity: Semantic ambiguity (0-1)
            
        Returns:
            Decision time in seconds
        """
        if ambiguity > 0:
            # Extended model with ambiguity
            n_effective = n_choices * (1 + ambiguity) ** 2
        else:
            n_effective = n_choices
        
        return self.a + self.b * np.log2(max(n_effective, 1))
    
    def effective_choices(self, n_nominal: int, ambiguity: float) -> float:
        """
        Calculate effective number of choices perceived by operator
        
        n_eff = n_nominal * (1 + ψ)²
        
        Args:
            n_nominal: Actual number of choices
            ambiguity: Semantic ambiguity
            
        Returns:
            Effective choice count
        """
        return n_nominal * (1 + ambiguity) ** 2


class CognitiveLoadModel:
    """
    Cognitive Load Theory Model
    
    Total Load = Intrinsic + Extraneous + Germane
    """
    
    def __init__(self, max_load: float = 100.0):
        """
        Initialize cognitive load model
        
        Args:
            max_load: Maximum cognitive capacity
        """
        self.max_load = max_load
    
    def calculate_load(self, intrinsic: float, extraneous: float, 
                       germane: float) -> float:
        """
        Calculate total cognitive load
        
        Args:
            intrinsic: Task inherent complexity
            extraneous: Poor design, confusing interface
            germane: Learning, schema formation
            
        Returns:
            Total cognitive load (0-max_load)
        """
        total = intrinsic + extraneous + germane
        return min(total, self.max_load)
    
    def load_from_factors(self, task_complexity: float, 
                         interface_quality: float,
                         n_alerts: int, 
                         ambiguity: float) -> float:
        """
        Estimate load from system factors
        
        Args:
            task_complexity: Task difficulty (0-1)
            interface_quality: UI quality (0-1, higher is better)
            n_alerts: Number of active alerts
            ambiguity: Semantic ambiguity (0-1)
            
        Returns:
            Estimated cognitive load
        """
        intrinsic = task_complexity * 40
        extraneous = (1 - interface_quality) * 30 + n_alerts * 5 + ambiguity * 20
        germane = 10  # Baseline learning load
        
        return self.calculate_load(intrinsic, extraneous, germane)


class SignalDetectionTheory:
    """
    Signal Detection Theory for anomaly detection tasks
    
    Sensitivity: d' = Z(Hit Rate) - Z(False Alarm Rate)
    Response Bias: β = f_s(λ) / f_n(λ)
    """
    
    def __init__(self):
        """Initialize SDT model"""
        pass
    
    def sensitivity(self, hit_rate: float, false_alarm_rate: float) -> float:
        """
        Calculate d' (d-prime) sensitivity
        
        Args:
            hit_rate: P(detect | signal present)
            false_alarm_rate: P(detect | noise only)
            
        Returns:
            d' sensitivity measure
        """
        from scipy.stats import norm
        
        # Clip to avoid edge cases
        hr = np.clip(hit_rate, 0.01, 0.99)
        far = np.clip(false_alarm_rate, 0.01, 0.99)
        
        d_prime = norm.ppf(hr) - norm.ppf(far)
        return d_prime
    
    def response_bias(self, hit_rate: float, false_alarm_rate: float) -> float:
        """
        Calculate β (beta) response bias
        
        β > 1: Conservative (avoid false alarms)
        β < 1: Liberal (avoid misses)
        β = 1: Neutral
        
        Args:
            hit_rate: P(detect | signal)
            false_alarm_rate: P(detect | noise)
            
        Returns:
            β bias measure
        """
        from scipy.stats import norm
        
        hr = np.clip(hit_rate, 0.01, 0.99)
        far = np.clip(false_alarm_rate, 0.01, 0.99)
        
        z_hr = norm.ppf(hr)
        z_far = norm.ppf(far)
        
        beta = np.exp((z_far ** 2 - z_hr ** 2) / 2)
        return beta


class YerkesDodsonModel:
    """
    Yerkes-Dodson Law: Performance vs. Stress (inverted U)
    
    P(stress) = A * exp(-(S - S₀)² / (2σ²))
    """
    
    def __init__(self, peak_performance: float = 100.0, 
                 optimal_stress: float = 0.5, 
                 stress_variance: float = 0.1):
        """
        Initialize Yerkes-Dodson model
        
        Args:
            peak_performance: Maximum performance level
            optimal_stress: Stress level for peak performance
            stress_variance: Width of performance curve
        """
        self.A = peak_performance
        self.S0 = optimal_stress
        self.sigma = stress_variance
    
    def performance(self, stress: float) -> float:
        """
        Calculate performance at given stress level
        
        Args:
            stress: Stress level (0-1)
            
        Returns:
            Performance level
        """
        return self.A * np.exp(-(stress - self.S0) ** 2 / (2 * self.sigma ** 2))
    
    def optimal_stress_range(self, threshold: float = 0.9) -> Tuple[float, float]:
        """
        Find stress range where performance > threshold * peak
        
        Args:
            threshold: Performance threshold (0-1)
            
        Returns:
            (lower_bound, upper_bound) of optimal stress
        """
        # Solve: threshold * A = A * exp(-(S - S0)²/(2σ²))
        delta = self.sigma * np.sqrt(-2 * np.log(threshold))
        return (self.S0 - delta, self.S0 + delta)


class FatigueAccumulationModel:
    """
    Decision fatigue accumulation over time
    
    F(t) = F(t-1) + k₁ * d(t) - k₂ * r(t)
    """
    
    def __init__(self, k1: float = 0.05, k2: float = 0.08):
        """
        Initialize fatigue model
        
        Args:
            k1: Fatigue accumulation rate
            k2: Recovery rate
        """
        self.k1 = k1
        self.k2 = k2
        self.fatigue = 0.0
        self.history = [0.0]
    
    def update(self, decision_demand: float, recovery: float = 0.0) -> float:
        """
        Update fatigue level
        
        Args:
            decision_demand: Mental demand from decisions
            recovery: Rest or break time
            
        Returns:
            Updated fatigue level
        """
        self.fatigue = max(0, self.fatigue + self.k1 * decision_demand - self.k2 * recovery)
        self.history.append(self.fatigue)
        return self.fatigue
    
    def performance_penalty(self) -> float:
        """
        Calculate performance degradation due to fatigue
        
        Returns:
            Multiplier (0-1)
        """
        return np.exp(-0.01 * self.fatigue)


class CognitiveErrorModel:
    """
    Probabilistic cognitive error model
    
    P(error) = 1 / (1 + exp(-a * CL(t) + b))
    """
    
    def __init__(self, a: float = 0.08, b: float = 5.0):
        """
        Initialize error model
        
        Args:
            a: Sensitivity to cognitive load
            b: Baseline error threshold
        """
        self.a = a
        self.b = b
    
    def error_probability(self, cognitive_load: float, fatigue: float = 0.0) -> float:
        """
        Calculate probability of human error
        
        Args:
            cognitive_load: Current cognitive load (0-100)
            fatigue: Fatigue level (0-100)
            
        Returns:
            Error probability (0-1)
        """
        # Combined effect of load and fatigue
        combined_load = cognitive_load + 0.5 * fatigue
        
        p_error = 1.0 / (1.0 + np.exp(-self.a * combined_load + self.b))
        return np.clip(p_error, 0.0, 1.0)


class AttentionAllocationModel:
    """
    Attention distribution across multiple tasks
    
    A_i(t) = w_i(t) / Σ w_j(t)
    """
    
    def __init__(self, n_tasks: int):
        """
        Initialize attention model
        
        Args:
            n_tasks: Number of concurrent tasks
        """
        self.n_tasks = n_tasks
        self.weights = np.ones(n_tasks) / n_tasks
    
    def allocate_attention(self, task_weights: np.ndarray) -> np.ndarray:
        """
        Calculate attention allocation
        
        Args:
            task_weights: Importance/urgency of each task
            
        Returns:
            Attention proportions (sums to 1)
        """
        attention = task_weights / task_weights.sum()
        return attention
    
    def update_weights(self, priorities: np.ndarray, learning_rate: float = 0.1):
        """
        Update task weights based on new priorities
        
        Args:
            priorities: New priority values
            learning_rate: Update rate
        """
        self.weights = (1 - learning_rate) * self.weights + learning_rate * priorities
        self.weights /= self.weights.sum()


class IntegratedCognitiveModel:
    """
    Integrated model combining multiple cognitive factors
    """
    
    def __init__(self):
        """Initialize integrated model"""
        self.hick_hyman = HickHymanModel()
        self.load_model = CognitiveLoadModel()
        self.fatigue_model = FatigueAccumulationModel()
        self.error_model = CognitiveErrorModel()
        self.yd_model = YerkesDodsonModel()
        
    def evaluate_operator_state(self, 
                                task_complexity: float,
                                n_choices: int,
                                ambiguity: float,
                                n_alerts: int,
                                stress: float) -> CognitiveState:
        """
        Comprehensive cognitive state evaluation
        
        Args:
            task_complexity: Task difficulty (0-1)
            n_choices: Decision alternatives
            ambiguity: Semantic ambiguity (0-1)
            n_alerts: Active system alerts
            stress: Operator stress level (0-1)
            
        Returns:
            CognitiveState object
        """
        # Calculate cognitive load
        workload = self.load_model.load_from_factors(
            task_complexity=task_complexity,
            interface_quality=0.7,  # Assume decent interface
            n_alerts=n_alerts,
            ambiguity=ambiguity
        )
        
        # Get current fatigue
        fatigue = self.fatigue_model.fatigue
        
        # Calculate attention capacity (reduced by fatigue)
        attention = 1.0 - (fatigue / 100.0) * 0.5
        
        # Performance under stress
        performance = self.yd_model.performance(stress) / 100.0
        
        return CognitiveState(
            workload=workload,
            fatigue=fatigue,
            attention=attention * performance,
            stress=stress
        )


if __name__ == "__main__":
    print("Cognitive Process Mathematics Test")
    print("=" * 50)
    
    # Test Hick-Hyman
    print("\n1. Hick-Hyman Law")
    hh = HickHymanModel()
    for n in [2, 4, 8]:
        t_clean = hh.decision_time(n, ambiguity=0.0)
        t_ambiguous = hh.decision_time(n, ambiguity=0.3)
        print(f"  n={n}: Clean={t_clean:.3f}s, Ambiguous={t_ambiguous:.3f}s")
    
    # Test Cognitive Load
    print("\n2. Cognitive Load")
    cl = CognitiveLoadModel()
    load = cl.load_from_factors(task_complexity=0.7, interface_quality=0.6, 
                                 n_alerts=5, ambiguity=0.3)
    print(f"  Total cognitive load: {load:.1f}/100")
    
    # Test Signal Detection
    print("\n3. Signal Detection Theory")
    sdt = SignalDetectionTheory()
    d_prime = sdt.sensitivity(hit_rate=0.85, false_alarm_rate=0.15)
    beta = sdt.response_bias(hit_rate=0.85, false_alarm_rate=0.15)
    print(f"  Sensitivity (d'): {d_prime:.2f}")
    print(f"  Response bias (β): {beta:.2f}")
    
    # Test Yerkes-Dodson
    print("\n4. Yerkes-Dodson Law")
    yd = YerkesDodsonModel()
    for stress in [0.2, 0.5, 0.8]:
        perf = yd.performance(stress)
        print(f"  Stress={stress:.1f}: Performance={perf:.1f}")
    
    # Test integrated model
    print("\n5. Integrated Cognitive Model")
    icm = IntegratedCognitiveModel()
    state = icm.evaluate_operator_state(
        task_complexity=0.6,
        n_choices=4,
        ambiguity=0.25,
        n_alerts=3,
        stress=0.5
    )
    print(f"  Workload: {state.workload:.1f}")
    print(f"  Fatigue: {state.fatigue:.1f}")
    print(f"  Attention: {state.attention:.2f}")
    print(f"  Stress: {state.stress:.2f}")
