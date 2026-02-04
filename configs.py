"""
Example configurations for different Rayleigh-Bénard convection test cases.
These can be used to reproduce standard benchmarks and validate the solvers.
"""

# Standard test cases from literature
TEST_CASES = {
    "weak_convection": {
        "name": "Weak Convection (Benchmark)",
        "Ra": 1e3,
        "Pr": 0.71,
        "aspect_ratio": 2.0,
        "nx": 40,
        "ny": 40,
        "max_iterations": 1000,
        "tolerance": 1e-5,
        "pinn_epochs": 3000,
        "hidden_layers": 4,
        "neurons_per_layer": 50,
        "learning_rate": 1e-3,
        "description": "Onset of convection, stable steady rolls"
    },
    
    "moderate_convection": {
        "name": "Moderate Convection",
        "Ra": 1e4,
        "Pr": 0.71,
        "aspect_ratio": 2.0,
        "nx": 50,
        "ny": 50,
        "max_iterations": 1500,
        "tolerance": 1e-5,
        "pinn_epochs": 5000,
        "hidden_layers": 5,
        "neurons_per_layer": 60,
        "learning_rate": 8e-4,
        "description": "Well-developed convection cells"
    },
    
    "strong_convection": {
        "name": "Strong Convection",
        "Ra": 1e5,
        "Pr": 0.71,
        "aspect_ratio": 2.0,
        "nx": 80,
        "ny": 80,
        "max_iterations": 2000,
        "tolerance": 1e-6,
        "pinn_epochs": 8000,
        "hidden_layers": 6,
        "neurons_per_layer": 80,
        "learning_rate": 5e-4,
        "description": "Complex flow patterns, multiple cells"
    },
    
    "air_ra1e4": {
        "name": "Air at Ra=10^4",
        "Ra": 1e4,
        "Pr": 0.71,  # Air
        "aspect_ratio": 1.0,
        "nx": 50,
        "ny": 50,
        "max_iterations": 1500,
        "tolerance": 1e-5,
        "pinn_epochs": 5000,
        "hidden_layers": 4,
        "neurons_per_layer": 50,
        "learning_rate": 1e-3,
        "description": "Air convection, square domain"
    },
    
    "water_ra1e5": {
        "name": "Water at Ra=10^5",
        "Ra": 1e5,
        "Pr": 7.0,  # Water
        "aspect_ratio": 2.0,
        "nx": 60,
        "ny": 60,
        "max_iterations": 2000,
        "tolerance": 1e-6,
        "pinn_epochs": 8000,
        "hidden_layers": 5,
        "neurons_per_layer": 70,
        "learning_rate": 5e-4,
        "description": "Water convection, elongated domain"
    },
    
    "mercury_ra1e4": {
        "name": "Mercury at Ra=10^4",
        "Ra": 1e4,
        "Pr": 0.025,  # Mercury (liquid metal)
        "aspect_ratio": 2.0,
        "nx": 50,
        "ny": 50,
        "max_iterations": 1500,
        "tolerance": 1e-5,
        "pinn_epochs": 6000,
        "hidden_layers": 5,
        "neurons_per_layer": 60,
        "learning_rate": 8e-4,
        "description": "Low Pr liquid metal convection"
    },
    
    "oil_ra1e4": {
        "name": "Oil at Ra=10^4",
        "Ra": 1e4,
        "Pr": 100.0,  # High viscosity oil
        "aspect_ratio": 2.0,
        "nx": 50,
        "ny": 50,
        "max_iterations": 2000,
        "tolerance": 1e-5,
        "pinn_epochs": 7000,
        "hidden_layers": 5,
        "neurons_per_layer": 60,
        "learning_rate": 5e-4,
        "description": "High Pr viscous fluid convection"
    },
    
    "wide_domain": {
        "name": "Wide Domain (AR=4)",
        "Ra": 5e4,
        "Pr": 0.71,
        "aspect_ratio": 4.0,
        "nx": 100,
        "ny": 50,
        "max_iterations": 2000,
        "tolerance": 1e-5,
        "pinn_epochs": 10000,
        "hidden_layers": 6,
        "neurons_per_layer": 80,
        "learning_rate": 5e-4,
        "description": "Multiple convection cells in wide domain"
    },
    
    "tall_domain": {
        "name": "Tall Domain (AR=0.5)",
        "Ra": 1e4,
        "Pr": 0.71,
        "aspect_ratio": 0.5,
        "nx": 25,
        "ny": 50,
        "max_iterations": 1500,
        "tolerance": 1e-5,
        "pinn_epochs": 5000,
        "hidden_layers": 4,
        "neurons_per_layer": 50,
        "learning_rate": 1e-3,
        "description": "Vertical convection in narrow domain"
    },
    
    "high_resolution": {
        "name": "High Resolution Test",
        "Ra": 5e4,
        "Pr": 0.71,
        "aspect_ratio": 2.0,
        "nx": 100,
        "ny": 100,
        "max_iterations": 3000,
        "tolerance": 1e-6,
        "pinn_epochs": 15000,
        "hidden_layers": 8,
        "neurons_per_layer": 100,
        "learning_rate": 3e-4,
        "description": "High-fidelity simulation for publication"
    }
}

# Validation targets from literature (where available)
VALIDATION_TARGETS = {
    "Ra_1e4_Pr_0.71": {
        "Nusselt_number": 2.2,  # Approximate from DNS
        "critical_Ra": 1708,  # Theoretical onset
        "description": "Standard air convection benchmark"
    }
}

# Recommended parameter sweeps for generalization testing
PARAMETER_SWEEPS = {
    "Ra_sweep": {
        "Ra_values": [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6],
        "Pr": 0.71,
        "aspect_ratio": 2.0,
        "purpose": "Test generalization across Rayleigh number"
    },
    
    "Pr_sweep": {
        "Ra": 1e4,
        "Pr_values": [0.025, 0.1, 0.71, 7.0, 100.0],
        "aspect_ratio": 2.0,
        "purpose": "Test generalization across Prandtl number"
    },
    
    "AR_sweep": {
        "Ra": 1e4,
        "Pr": 0.71,
        "aspect_ratio_values": [0.5, 1.0, 2.0, 3.0, 4.0],
        "purpose": "Test generalization across aspect ratios"
    }
}

# PINN architecture experiments
ARCHITECTURE_EXPERIMENTS = {
    "shallow_network": {
        "hidden_layers": 2,
        "neurons_per_layer": 30,
        "description": "Minimal network capacity"
    },
    
    "standard_network": {
        "hidden_layers": 4,
        "neurons_per_layer": 50,
        "description": "Baseline configuration"
    },
    
    "deep_network": {
        "hidden_layers": 8,
        "neurons_per_layer": 50,
        "description": "Deeper architecture"
    },
    
    "wide_network": {
        "hidden_layers": 4,
        "neurons_per_layer": 100,
        "description": "Wider layers"
    },
    
    "deep_and_wide": {
        "hidden_layers": 8,
        "neurons_per_layer": 100,
        "description": "Maximum capacity (slow training)"
    }
}

# Expected computation times (approximate, hardware-dependent)
EXPECTED_TIMES = {
    "traditional_solver": {
        "30x30_grid": "10-30 seconds",
        "50x50_grid": "30-90 seconds",
        "100x100_grid": "2-5 minutes"
    },
    
    "pinn_solver": {
        "2000_epochs": "30-60 seconds",
        "5000_epochs": "1-3 minutes",
        "10000_epochs": "3-8 minutes",
        "20000_epochs": "8-20 minutes"
    }
}

# Tips for successful training
TRAINING_TIPS = {
    "convergence": [
        "Monitor training loss - should decrease steadily",
        "If loss plateaus early, try higher learning rate",
        "If loss oscillates, try lower learning rate",
        "Check boundary loss vs physics loss separately"
    ],
    
    "architecture": [
        "Start with 4 layers, 50 neurons baseline",
        "Add depth for complex patterns",
        "Add width for high-frequency features",
        "Deeper is better for smooth solutions"
    ],
    
    "hyperparameters": [
        "Learning rate: 1e-3 is good starting point",
        "Boundary weight: 10x physics loss typically works",
        "Training points: 2000 interior + 500 boundary is sufficient",
        "Epochs: scale with problem difficulty"
    ],
    
    "debugging": [
        "Check if traditional solver converged first",
        "Verify boundary conditions are enforced",
        "Plot loss components separately",
        "Test on simpler cases (lower Ra) first"
    ]
}

if __name__ == "__main__":
    print("Available test cases:")
    for key, config in TEST_CASES.items():
        print(f"\n{key}:")
        print(f"  Name: {config['name']}")
        print(f"  Ra: {config['Ra']:.1e}, Pr: {config['Pr']}")
        print(f"  Description: {config['description']}")
