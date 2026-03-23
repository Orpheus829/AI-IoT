"""
Discrete Event Simulation for AIoT Work Systems
Validates mathematical framework from Chapter 13
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys
sys.path.append('/home/claude/aiot_wsd_project/modules')
from modules.configuration import config, semantic_efficiency, effective_service_rate, system_latency

@dataclass
class SimulationResult:
    """Results from a simulation run"""
    ambiguity: float
    effective_service_rate: float
    avg_latency: float
    system_status: str
    cognitive_load: float
    decision_time: float

class AIoTWorkSystemSimulation:
    """
    Discrete Event Simulation of AIoT work system
    Models the Semantic-Latency coupling from Chapter 12
    """
    
    def __init__(self, 
                 arrival_rate: float = None,
                 base_service_rate: float = None,
                 cognitive_coeff: float = None):
        """
        Initialize simulation
        
        Args:
            arrival_rate: λ (tasks/hour)
            base_service_rate: μ_base (tasks/hour)
            cognitive_coeff: κ (cognitive resistance)
        """
        self.lambda_ = arrival_rate or config.TASK_ARRIVAL_RATE
        self.mu_base = base_service_rate or config.BASE_SERVICE_RATE
        self.kappa = cognitive_coeff or config.COGNITIVE_RESISTANCE_COEFF
        
        # Simulation state
        self.current_time = 0.0
        self.queue = []
        self.completed_tasks = 0
        self.total_wait_time = 0.0
        
    def generate_arrival_time(self) -> float:
        """Generate next task arrival (Poisson process)"""
        return np.random.exponential(1.0 / self.lambda_)
    
    def generate_service_time(self, ambiguity: float) -> float:
        """
        Generate service time based on ambiguity
        
        Service rate depends on semantic clarity:
        μ_eff = μ_base * exp(-κψ)
        """
        mu_eff = effective_service_rate(ambiguity, self.mu_base)
        return np.random.exponential(1.0 / mu_eff)
    
    def run_single_scenario(self, ambiguity: float, 
                           duration_hours: int = 8,
                           n_iterations: int = 1000) -> SimulationResult:
        """
        Run simulation for specific ambiguity level
        
        Args:
            ambiguity: Semantic ambiguity (0-1)
            duration_hours: Simulation duration
            n_iterations: Number of runs for statistics
            
        Returns:
            SimulationResult object
        """
        latencies = []
        service_rates = []
        
        for _ in range(n_iterations):
            self.current_time = 0.0
            self.queue = []
            wait_times = []
            
            # Generate events
            arrivals = []
            t = 0
            while t < duration_hours:
                t += self.generate_arrival_time()
                if t < duration_hours:
                    arrivals.append(t)
            
            # Process events
            for arrival_time in arrivals:
                # Task arrives
                service_time = self.generate_service_time(ambiguity)
                
                # Check if server busy
                if len(self.queue) == 0:
                    # Start immediately
                    complete_time = arrival_time + service_time
                    wait_time = 0
                else:
                    # Wait in queue
                    last_complete = self.queue[-1][1]
                    start_time = max(arrival_time, last_complete)
                    complete_time = start_time + service_time
                    wait_time = start_time - arrival_time
                
                self.queue.append((arrival_time, complete_time))
                wait_times.append(wait_time)
            
            if wait_times:
                latencies.append(np.mean(wait_times))
                service_rates.append(1.0 / np.mean([ct - at for at, ct in self.queue]))
            
        # Aggregate results
        avg_latency = np.mean(latencies) * 3600  # Convert to seconds
        avg_service_rate = np.mean(service_rates) if service_rates else 0
        
        # Determine system status
        mu_eff = effective_service_rate(ambiguity, self.mu_base)
        if mu_eff <= self.lambda_:
            status = "Unstable"
        elif avg_latency > 1000:
            status = "Critical"
        elif avg_latency > 500:
            status = "Stressed"
        else:
            status = "Stable"
        
        # Calculate cognitive metrics
        from modules.cognitive_model import HickHymanModel, CognitiveLoadModel
        hh = HickHymanModel()
        cl = CognitiveLoadModel()
        
        decision_time = hh.decision_time(n_choices=4, ambiguity=ambiguity)
        cognitive_load = cl.load_from_factors(
            task_complexity=0.6,
            interface_quality=0.7,
            n_alerts=int(avg_latency / 100),
            ambiguity=ambiguity
        )
        
        return SimulationResult(
            ambiguity=ambiguity,
            effective_service_rate=mu_eff,
            avg_latency=avg_latency,
            system_status=status,
            cognitive_load=cognitive_load,
            decision_time=decision_time
        )
    
    def run_ambiguity_sweep(self, 
                           ambiguity_levels: List[float] = None,
                           iterations: int = 1000) -> pd.DataFrame:
        """
        Run simulation across multiple ambiguity levels
        Reproduces Table 13.1 from paper
        
        Args:
            ambiguity_levels: List of ψ values to test
            iterations: Samples per level
            
        Returns:
            DataFrame with results
        """
        if ambiguity_levels is None:
            ambiguity_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
        
        results = []
        
        print("Running Ambiguity Sweep Simulation...")
        for ambiguity in ambiguity_levels:
            print(f"  Testing ψ = {ambiguity:.2f}...")
            result = self.run_single_scenario(ambiguity, 
                                             duration_hours=8,
                                             n_iterations=iterations)
            results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'Semantic Ambiguity (ψ)': r.ambiguity,
                'Effective Service Rate (μ_eff)': r.effective_service_rate,
                'Avg System Latency (sec)': r.avg_latency,
                'System Status': r.system_status,
                'Cognitive Load': r.cognitive_load,
                'Decision Time (sec)': r.decision_time
            }
            for r in results
        ])
        
        return df


class CognitiveWorkloadSimulation:
    """
    Simulate cognitive load reduction via NLP
    Reproduces Table 13.2 from paper
    """
    
    def __init__(self):
        """Initialize simulation"""
        from modules.cognitive_model import HickHymanModel, CognitiveLoadModel
        from modules.semantic_disambiguation import IntegratedWSDSystem, INDUSTRIAL_SENSE_INVENTORY, INDUSTRIAL_GLOSSES
        
        self.hh = HickHymanModel()
        self.cl = CognitiveLoadModel()
        self.wsd = IntegratedWSDSystem(INDUSTRIAL_SENSE_INVENTORY, INDUSTRIAL_GLOSSES)
    
    def simulate_baseline(self, n_commands: int = 100) -> Dict:
        """Baseline system without disambiguation"""
        ambiguities = []
        decision_times = []
        cognitive_loads = []
        
        # Simulate high-ambiguity commands
        for _ in range(n_commands):
            ambiguity = np.random.uniform(0.35, 0.55)  # High ambiguity
            n_choices = np.random.randint(4, 8)
            
            # Effective choices increase with ambiguity
            n_eff = self.hh.effective_choices(n_choices, ambiguity)
            dt = self.hh.decision_time(n_choices, ambiguity)
            cl = self.cl.load_from_factors(0.6, 0.5, 5, ambiguity)
            
            ambiguities.append(ambiguity)
            decision_times.append(dt)
            cognitive_loads.append(cl)
        
        return {
            'avg_ambiguity': np.mean(ambiguities),
            'avg_effective_choices': np.mean([self.hh.effective_choices(5, a) for a in ambiguities]),
            'avg_decision_time': np.mean(decision_times),
            'avg_cognitive_load': np.mean(cognitive_loads)
        }
    
    def simulate_integrated(self, n_commands: int = 100) -> Dict:
        """Integrated system with WSD"""
        test_commands = [
            "check the assembly line",
            "stop the robot arm",
            "reduce load on system",
            "inspect electrical line"
        ]
        
        ambiguities = []
        decision_times = []
        cognitive_loads = []
        
        for _ in range(n_commands):
            # Random command
            cmd = np.random.choice(test_commands)
            result = self.wsd.process_command(cmd)
            
            # WSD reduces ambiguity
            ambiguity = result['max_ambiguity']
            n_choices = len(result['disambiguated']) + 2
            
            dt = self.hh.decision_time(n_choices, ambiguity)
            cl = self.cl.load_from_factors(0.6, 0.8, 2, ambiguity)
            
            ambiguities.append(ambiguity)
            decision_times.append(dt)
            cognitive_loads.append(cl)
        
        return {
            'avg_ambiguity': np.mean(ambiguities),
            'avg_effective_choices': np.mean([self.hh.effective_choices(3, a) for a in ambiguities]),
            'avg_decision_time': np.mean(decision_times),
            'avg_cognitive_load': np.mean(cognitive_loads)
        }
    
    def compare_systems(self) -> pd.DataFrame:
        """
        Compare baseline vs integrated systems
        Reproduces Table 13.2
        """
        baseline = self.simulate_baseline(n_commands=200)
        integrated = self.simulate_integrated(n_commands=200)
        
        df = pd.DataFrame([
            {
                'System': 'Baseline (No WSD)',
                'Avg Ambiguity (ψ)': baseline['avg_ambiguity'],
                'Effective Choices': baseline['avg_effective_choices'],
                'Decision Time (sec)': baseline['avg_decision_time'],
                'Cognitive Load': baseline['avg_cognitive_load']
            },
            {
                'System': 'Integrated AIoT (With WSD)',
                'Avg Ambiguity (ψ)': integrated['avg_ambiguity'],
                'Effective Choices': integrated['avg_effective_choices'],
                'Decision Time (sec)': integrated['avg_decision_time'],
                'Cognitive Load': integrated['avg_cognitive_load']
            }
        ])
        
        # Calculate improvements
        df['Improvement %'] = 0.0
        for col in ['Avg Ambiguity (ψ)', 'Effective Choices', 'Decision Time (sec)', 'Cognitive Load']:
            improvement = (1 - df.loc[1, col] / df.loc[0, col]) * 100
            df.loc[1, f'{col} Improvement'] = improvement
        
        return df


if __name__ == "__main__":
    print("AIoT Work System Simulation")
    print("=" * 50)
    
    # Table 13.1: Latency vs Ambiguity
    print("\n### Table 13.1: System Latency vs. Semantic Ambiguity ###\n")
    sim = AIoTWorkSystemSimulation()
    results_df = sim.run_ambiguity_sweep(iterations=100)
    print(results_df.to_string(index=False))
    
    # Table 13.2: Cognitive Load Comparison
    print("\n\n### Table 13.2: Cognitive Load Comparison ###\n")
    cog_sim = CognitiveWorkloadSimulation()
    comparison_df = cog_sim.compare_systems()
    print(comparison_df.to_string(index=False))
