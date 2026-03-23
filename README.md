# AIoT Work System Design: Integrated Mathematical Framework

**A Stochastic, Cognitive, and Semantic Modeling Approach Using Semantic Disambiguation**

This project implements the complete mathematical framework from the research paper "Integrated Work System Design in AI–IoT Environments" by Gourab Pal (IIT Kharagpur).

## Project Overview

This implementation validates the theoretical framework through:
- Stochastic modeling (Markov chains, Weibull reliability, queueing theory)
- Cognitive process mathematics (Hick-Hyman Law, cognitive load, signal detection)
- Semantic disambiguation algorithms (Lesk, embedding-based, ensemble methods)
- Reinforcement learning for task allocation
- Discrete event simulation validating the semantic-latency coupling

## Key Mathematical Contributions

### 1. Semantic Efficiency Function (Eq. 12.1)
```
η(ψ) = e^(-κψ)
```
Quantifies how semantic ambiguity reduces operator efficiency.

### 2. Effective Service Rate (Eq. 12.2)
```
μ_eff(t) = μ_base * e^(-κψ(t))
```
Links NLP quality directly to production throughput.

### 3. Semantic Latency Equation (Eq. 12.3)
```
W_q(ψ) = λ / [μ_base * e^(-κψ) * (μ_base * e^(-κψ) - λ)]
```
Proves non-linear relationship between language clarity and factory output.

### 4. Critical Ambiguity Threshold (Eq. 12.4)
```
ψ_crit = (1/κ) * ln(μ_base / λ)
```
Identifies the tipping point where the system becomes unstable.

## Project Structure

```
aiot_wsd_project/
│
├── data/
│   └── ai4i2020.csv               # AI4I Predictive Maintenance Dataset
│
├── modules/
│   ├── config.py                  # System parameters & constants
│   ├── dataset_loader.py          # Data preprocessing
│   ├── kalman_filter.py           # Sensor fusion (Chapter 5)
│   ├── stochastic_engine.py       # Markov, Weibull, queueing (Chapter 8)
│   ├── cognitive_model.py         # Cognitive mathematics (Chapter 9)
│   ├── semantic_disambiguation.py # WSD algorithms (Chapter 10)
│   ├── rl_agent.py                # Q-learning task allocator (Chapter 11)
│   └── simulation.py              # DES validation (Chapter 13)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sensor_fusion.ipynb
│   ├── 03_stochastic_models.ipynb
│   ├── 04_cognitive_models.ipynb
│   ├── 05_wsd_evaluation.ipynb
│   ├── 06_simulation_results.ipynb
│   └── 07_integrated_pipeline.ipynb
│
├── results/
│   ├── table_13_1_latency.csv     # Simulation results (Table 13.1)
│   ├── table_13_2_cognitive.csv   # Cognitive comparison (Table 13.2)
│   └── figures/                   # Publication-quality plots
│
├── main.py                        # Complete pipeline
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Installation

```bash
cd aiot_wsd_project
pip install -r requirements.txt
```

## Usage

### Run Complete Pipeline
```bash
python main.py
```

This executes all 8 steps:
1. Data loading and preprocessing
2. Sensor fusion with Kalman filtering
3. Stochastic modeling
4. Cognitive process modeling
5. Semantic disambiguation
6. RL agent training
7. Discrete event simulation
8. Visualization generation

### Run Individual Modules
```python
# Test Kalman filter
from modules.kalman_filter import create_machine_health_kf
kf = create_machine_health_kf()

# Test Markov chain
from modules.stochastic_engine import create_machine_health_markov
mc = create_machine_health_markov()
steady_state = mc.steady_state()

# Test semantic disambiguation
from modules.semantic_disambiguation import IntegratedWSDSystem
from modules.semantic_disambiguation import INDUSTRIAL_SENSE_INVENTORY
wsd = IntegratedWSDSystem(INDUSTRIAL_SENSE_INVENTORY)
result = wsd.process_command("check the assembly line")

# Run simulation
from modules.simulation import AIoTWorkSystemSimulation
sim = AIoTWorkSystemSimulation()
results = sim.run_ambiguity_sweep()
```

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**
- Source: UCI ML Repository / Kaggle
- Size: 10,000 records
- Features: Air temp, process temp, rotational speed, torque, tool wear
- Targets: 5 failure modes (TWF, HDF, PWF, OSF, RNF)

If the dataset is not available, the system automatically generates synthetic data with the same structure.

## Key Results

### Table 13.1: System Latency vs. Semantic Ambiguity

| ψ    | μ_eff | Latency (sec) | Status   |
|------|-------|---------------|----------|
| 0.0  | 60.0  | ~180          | Stable   |
| 0.1  | 53.2  | ~272          | Stable   |
| 0.2  | 47.1  | ~507          | Stressed |
| 0.3  | 41.8  | ~1,998        | Critical |
| 0.4  | 37.1  | ∞             | Unstable |

### Table 13.2: Cognitive Load Comparison

| System           | Ambiguity | Decision Time | Workload | Improvement |
|------------------|-----------|---------------|----------|-------------|
| Baseline         | 0.45      | 3.8s          | 78/100   | -           |
| Integrated AIoT  | 0.08      | 1.9s          | 42/100   | 50% faster  |

## Mathematical Framework Validation

The simulation validates:
1. **Non-linearity**: Small improvements in NLP accuracy (ψ: 0.4→0.3) prevent system collapse
2. **Cognitive preservation**: WSD reduces decision time by 50%
3. **Safety-critical design**: Semantic interlock recommended at ψ > 0.35

## Citation

```bibtex
@article{pal2025aiot,
  title={Integrated Work System Design in AI–IoT Environments: A Stochastic, Cognitive, and Semantic Modelling Approach},
  author={Pal, Gourab},
  institution={Indian Institute of Technology, Kharagpur},
  department={Manufacturing Science and Engineering},
  year={2025},
}
```

## License

This project is for academic research and educational purposes.

## Author

**Gourab Pal (22MF3IM08)**  
Manufacturing Science and Engineering  
Indian Institute of Technology, Kharagpur  

## Contact

For questions about the implementation, please refer to the paper or examine the detailed code comments in each module.
