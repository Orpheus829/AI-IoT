"""
Stochastic Modeling Engine for AIoT Work Systems
Implements Markov chains, Weibull reliability, queueing models from Chapter 8
"""

import numpy as np
from scipy import stats
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class MarkovState:
    """Represents state in a Markov chain"""
    current_state: int
    probability_vector: np.ndarray
    
class MarkovChain:
    """
    Discrete-time Markov Chain for system state modeling
    
    State evolution: π(t+1) = π(t) * P
    Where P is the transition matrix
    """
    
    def __init__(self, transition_matrix: np.ndarray, state_names: Optional[List[str]] = None):
        """
        Initialize Markov chain
        
        Args:
            transition_matrix: Square stochastic matrix (rows sum to 1)
            state_names: Optional state labels
        """
        self.P = np.array(transition_matrix)
        self.n_states = self.P.shape[0]
        
        # Validate transition matrix
        assert self.P.shape[0] == self.P.shape[1], "Matrix must be square"
        assert np.allclose(self.P.sum(axis=1), 1.0), "Rows must sum to 1"
        
        self.state_names = state_names or [f"State_{i}" for i in range(self.n_states)]
        
        # Current state distribution
        self.pi = np.zeros(self.n_states)
        self.pi[0] = 1.0  # Start in first state
        
    def step(self, n_steps: int = 1) -> np.ndarray:
        """
        Evolve the chain forward n steps
        
        Args:
            n_steps: Number of time steps
            
        Returns:
            State probability vector after n steps
        """
        for _ in range(n_steps):
            self.pi = self.pi @ self.P
        return self.pi
    
    def steady_state(self, tolerance: float = 1e-10, max_iter: int = 10000) -> np.ndarray:
        """
        Find steady-state distribution
        
        Solves: π = π * P
        
        Returns:
            Steady-state probability vector
        """
        pi_old = self.pi.copy()
        
        for _ in range(max_iter):
            pi_new = pi_old @ self.P
            if np.allclose(pi_new, pi_old, atol=tolerance):
                self.pi = pi_new
                return pi_new
            pi_old = pi_new
        
        print("Warning: Steady state did not converge")
        return pi_old
    
    def simulate(self, n_steps: int, initial_state: int = 0) -> List[int]:
        """
        Simulate a sample path through the chain
        
        Args:
            n_steps: Length of simulation
            initial_state: Starting state index
            
        Returns:
            List of state indices
        """
        path = [initial_state]
        current = initial_state
        
        for _ in range(n_steps - 1):
            # Sample next state based on transition probabilities
            current = np.random.choice(self.n_states, p=self.P[current])
            path.append(current)
        
        return path
    
    def mean_first_passage_time(self, from_state: int, to_state: int) -> float:
        """
        Calculate expected time to reach to_state from from_state
        
        Returns:
            Mean first passage time
        """
        # Create absorbing chain
        Q = self.P.copy()
        Q[to_state, :] = 0
        Q[to_state, to_state] = 1
        
        # Remove absorbing state
        indices = [i for i in range(self.n_states) if i != to_state]
        Q_sub = Q[np.ix_(indices, indices)]
        
        # Fundamental matrix: N = (I - Q)^{-1}
        N = np.linalg.inv(np.eye(len(indices)) - Q_sub)
        
        # Mean first passage time
        from_idx = indices.index(from_state)
        return N[from_idx].sum()


class WeibullReliability:
    """
    Weibull distribution for reliability modeling
    
    R(t) = exp(-(t/β)^α)
    
    Where:
    - α (shape): Determines failure mode
      α < 1: Infant mortality
      α = 1: Random failures
      α > 1: Wear-out
    - β (scale): Characteristic life
    """
    
    def __init__(self, shape: float, scale: float):
        """
        Initialize Weibull model
        
        Args:
            shape: α parameter (failure mode)
            scale: β parameter (characteristic life)
        """
        self.alpha = shape
        self.beta = scale
        self.distribution = stats.weibull_min(c=shape, scale=scale)
    
    def reliability(self, t: float) -> float:
        """
        Calculate reliability at time t
        
        R(t) = P(T > t)
        
        Args:
            t: Time
            
        Returns:
            Probability of survival
        """
        return np.exp(-(t / self.beta) ** self.alpha)
    
    def hazard_rate(self, t: float) -> float:
        """
        Calculate instantaneous failure rate
        
        h(t) = f(t) / R(t)
        
        Args:
            t: Time
            
        Returns:
            Hazard rate
        """
        return (self.alpha / self.beta) * (t / self.beta) ** (self.alpha - 1)
    
    def mean_time_to_failure(self) -> float:
        """
        Calculate MTTF = E[T]
        
        Returns:
            Expected lifetime
        """
        from scipy.special import gamma
        return self.beta * gamma(1 + 1/self.alpha)
    
    def predict_failure_probability(self, t_current: float, t_horizon: float) -> float:
        """
        Probability of failure in [t_current, t_current + t_horizon]
        
        Returns:
            Failure probability
        """
        return self.reliability(t_current) - self.reliability(t_current + t_horizon)


class QueueingModel:
    """
    M/M/1 Queueing model for workflow analysis
    
    M/M/1: Poisson arrivals, Exponential service, 1 server
    """
    
    def __init__(self, arrival_rate: float, service_rate: float):
        """
        Initialize M/M/1 queue
        
        Args:
            arrival_rate: λ (arrivals per unit time)
            service_rate: μ (services per unit time)
        """
        self.lambda_ = arrival_rate
        self.mu = service_rate
        self.rho = arrival_rate / service_rate  # Utilization
        
        if self.rho >= 1.0:
            print(f"Warning: System unstable (ρ = {self.rho:.2f} >= 1)")
    
    def mean_queue_length(self) -> float:
        """
        Average number in queue (not including service)
        
        L_q = λ² / (μ(μ - λ))
        
        Returns:
            Mean queue length
        """
        if self.rho >= 1.0:
            return np.inf
        return (self.lambda_ ** 2) / (self.mu * (self.mu - self.lambda_))
    
    def mean_system_size(self) -> float:
        """
        Average number in system (queue + service)
        
        L = λ / (μ - λ)
        
        Returns:
            Mean system size
        """
        if self.rho >= 1.0:
            return np.inf
        return self.lambda_ / (self.mu - self.lambda_)
    
    def mean_waiting_time(self) -> float:
        """
        Average time in queue
        
        W_q = λ / (μ(μ - λ))
        
        Returns:
            Mean waiting time
        """
        if self.rho >= 1.0:
            return np.inf
        return self.lambda_ / (self.mu * (self.mu - self.lambda_))
    
    def mean_system_time(self) -> float:
        """
        Average time in system (wait + service)
        
        W = 1 / (μ - λ)
        
        Returns:
            Mean system time
        """
        if self.rho >= 1.0:
            return np.inf
        return 1.0 / (self.mu - self.lambda_)
    
    def utilization(self) -> float:
        """Server utilization ρ = λ/μ"""
        return self.rho


class PoissonProcess:
    """
    Poisson process for event modeling
    
    P(N(t) = k) = (λt)^k * e^(-λt) / k!
    """
    
    def __init__(self, rate: float):
        """
        Initialize Poisson process
        
        Args:
            rate: λ (events per unit time)
        """
        self.rate = rate
    
    def probability(self, n_events: int, time: float) -> float:
        """
        Probability of exactly n events in time interval
        
        Args:
            n_events: Number of events
            time: Time duration
            
        Returns:
            Probability
        """
        lambda_t = self.rate * time
        return (lambda_t ** n_events) * np.exp(-lambda_t) / np.math.factorial(n_events)
    
    def simulate(self, duration: float) -> Tuple[np.ndarray, int]:
        """
        Simulate event arrivals
        
        Args:
            duration: Simulation time
            
        Returns:
            (event_times, n_events)
        """
        events = []
        t = 0
        
        while t < duration:
            # Exponential inter-arrival time
            t += np.random.exponential(1.0 / self.rate)
            if t < duration:
                events.append(t)
        
        return np.array(events), len(events)


class StochasticWorkload:
    """
    Stochastic model of human workload evolution
    
    w(t+1) = w(t) + ε(t)
    
    Where ε ~ N(0, σ²)
    """
    
    def __init__(self, initial_load: float = 50.0, noise_std: float = 5.0):
        """
        Initialize workload model
        
        Args:
            initial_load: Starting workload
            noise_std: Standard deviation of random fluctuations
        """
        self.w = initial_load
        self.sigma = noise_std
        self.history = [initial_load]
    
    def step(self, task_demand: float = 0.0, recovery: float = 0.0) -> float:
        """
        Evolve workload one step
        
        Args:
            task_demand: Additional load from new task
            recovery: Load reduction (break, automation)
            
        Returns:
            Updated workload
        """
        # Random fluctuation
        noise = np.random.normal(0, self.sigma)
        
        # Update equation
        self.w = np.clip(self.w + task_demand - recovery + noise, 0, 100)
        self.history.append(self.w)
        
        return self.w
    
    def simulate(self, n_steps: int, task_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simulate workload trajectory
        
        Args:
            n_steps: Number of time steps
            task_profile: Array of task demands over time
            
        Returns:
            Workload time series
        """
        if task_profile is None:
            task_profile = np.random.uniform(5, 15, n_steps)
        
        trajectory = []
        for i in range(n_steps):
            self.step(task_demand=task_profile[i])
            trajectory.append(self.w)
        
        return np.array(trajectory)


def create_machine_health_markov() -> MarkovChain:
    """
    Create Markov chain for machine health states
    States: Healthy, Degrading, Failed
    """
    P = np.array([
        [0.95, 0.04, 0.01],  # Healthy -> [H, D, F]
        [0.20, 0.70, 0.10],  # Degrading -> [H, D, F]
        [0.00, 0.00, 1.00]   # Failed -> [H, D, F] (absorbing)
    ])
    
    return MarkovChain(P, state_names=['Healthy', 'Degrading', 'Failed'])


if __name__ == "__main__":
    print("Stochastic Modeling Engine Test")
    print("=" * 50)
    
    # Test Markov chain
    print("\n1. Markov Chain - Machine Health")
    mc = create_machine_health_markov()
    steady = mc.steady_state()
    print(f"Steady-state probabilities:")
    for i, name in enumerate(mc.state_names):
        print(f"  {name}: {steady[i]:.4f}")
    
    # Test Weibull
    print("\n2. Weibull Reliability")
    weibull = WeibullReliability(shape=2.5, scale=1000)
    print(f"R(500 hours) = {weibull.reliability(500):.4f}")
    print(f"MTTF = {weibull.mean_time_to_failure():.2f} hours")
    print(f"Hazard rate at 800 hours = {weibull.hazard_rate(800):.6f}")
    
    # Test queueing
    print("\n3. M/M/1 Queue")
    queue = QueueingModel(arrival_rate=40, service_rate=60)
    print(f"Utilization: {queue.utilization():.2f}")
    print(f"Mean queue length: {queue.mean_queue_length():.2f}")
    print(f"Mean waiting time: {queue.mean_waiting_time():.4f} hours")
    
    # Test Poisson
    print("\n4. Poisson Process")
    poisson = PoissonProcess(rate=5.0)  # 5 events/hour
    events, n = poisson.simulate(duration=10)
    print(f"Simulated {n} events in 10 hours")
    print(f"P(N(1) = 5) = {poisson.probability(5, 1.0):.4f}")
