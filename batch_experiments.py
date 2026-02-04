"""
Batch Experiment Runner for Convection Solver
This script runs systematic experiments for research and validation purposes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
import os
from configs import TEST_CASES, PARAMETER_SWEEPS

# Import solver classes (assuming they're in app.py)
# In practice, you'd refactor these into separate modules
# from app import TraditionalSolver, PINNSolver

class ExperimentRunner:
    """Run and log systematic experiments"""
    
    def __init__(self, output_dir='experiments'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(f"{output_dir}/results", exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)
        os.makedirs(f"{output_dir}/data", exist_ok=True)
        
        self.results = []
        
    def run_single_experiment(self, config, exp_name):
        """Run a single experiment with given configuration"""
        
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp_name}")
        print(f"{'='*60}")
        print(f"Ra={config['Ra']:.1e}, Pr={config['Pr']}, AR={config['aspect_ratio']}")
        
        result = {
            'experiment': exp_name,
            'timestamp': datetime.now().isoformat(),
            'config': config.copy()
        }
        
        try:
            # Run traditional solver
            print("\n[1/2] Running traditional solver...")
            start_time = time.time()
            
            # TODO: Uncomment when integrated with actual solvers
            # trad_solver = TraditionalSolver(
            #     config['nx'], config['ny'], 
            #     config['Ra'], config['Pr'], config['aspect_ratio'],
            #     config['max_iterations'], config['tolerance']
            # )
            # trad_results = trad_solver.solve()
            
            trad_time = time.time() - start_time
            
            result['traditional'] = {
                'time': trad_time,
                # 'converged': True,
                # 'iterations': trad_solver.iterations
            }
            
            print(f"✓ Traditional solver completed in {trad_time:.2f}s")
            
            # Run PINN solver
            print("\n[2/2] Running PINN solver...")
            start_time = time.time()
            
            # TODO: Uncomment when integrated
            # layers = [2] + [config['neurons_per_layer']] * config['hidden_layers'] + [4]
            # pinn_solver = PINNSolver(
            #     config['nx'], config['ny'],
            #     config['Ra'], config['Pr'], config['aspect_ratio'],
            #     layers, config['pinn_epochs'], config['learning_rate']
            # )
            # loss_history = pinn_solver.train()
            
            pinn_time = time.time() - start_time
            
            result['pinn'] = {
                'time': pinn_time,
                # 'final_loss': loss_history[-1],
                # 'epochs': config['pinn_epochs']
            }
            
            print(f"✓ PINN solver completed in {pinn_time:.2f}s")
            
            # Compute error metrics
            # TODO: Compute actual errors
            result['errors'] = {
                'temperature_l2': 0.0,  # Placeholder
                'velocity_l2': 0.0,
                'max_error': 0.0
            }
            
            result['success'] = True
            
        except Exception as e:
            print(f"✗ Experiment failed: {str(e)}")
            result['success'] = False
            result['error'] = str(e)
        
        self.results.append(result)
        return result
    
    def run_test_suite(self, test_names=None):
        """Run a suite of test cases"""
        
        if test_names is None:
            test_names = list(TEST_CASES.keys())
        
        print(f"\n🧪 Running test suite with {len(test_names)} cases")
        print("="*60)
        
        for test_name in test_names:
            config = TEST_CASES[test_name]
            self.run_single_experiment(config, test_name)
        
        self.save_results()
        self.generate_report()
    
    def run_parameter_sweep(self, sweep_name):
        """Run a parameter sweep experiment"""
        
        if sweep_name not in PARAMETER_SWEEPS:
            print(f"Error: Unknown sweep '{sweep_name}'")
            return
        
        sweep_config = PARAMETER_SWEEPS[sweep_name]
        print(f"\n🔬 Running parameter sweep: {sweep_name}")
        print(f"Purpose: {sweep_config['purpose']}")
        print("="*60)
        
        # Determine which parameter to sweep
        if 'Ra_values' in sweep_config:
            param_name = 'Ra'
            param_values = sweep_config['Ra_values']
            base_config = TEST_CASES['moderate_convection'].copy()
            base_config['Pr'] = sweep_config['Pr']
            base_config['aspect_ratio'] = sweep_config['aspect_ratio']
            
        elif 'Pr_values' in sweep_config:
            param_name = 'Pr'
            param_values = sweep_config['Pr_values']
            base_config = TEST_CASES['moderate_convection'].copy()
            base_config['Ra'] = sweep_config['Ra']
            base_config['aspect_ratio'] = sweep_config['aspect_ratio']
            
        elif 'aspect_ratio_values' in sweep_config:
            param_name = 'aspect_ratio'
            param_values = sweep_config['aspect_ratio_values']
            base_config = TEST_CASES['moderate_convection'].copy()
            base_config['Ra'] = sweep_config['Ra']
            base_config['Pr'] = sweep_config['Pr']
        
        # Run experiments
        for value in param_values:
            config = base_config.copy()
            config[param_name] = value
            exp_name = f"{sweep_name}_{param_name}_{value}"
            
            self.run_single_experiment(config, exp_name)
        
        self.save_results()
        self.plot_sweep_results(sweep_name, param_name, param_values)
    
    def save_results(self):
        """Save experiment results to JSON and CSV"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results as JSON
        json_path = f"{self.output_dir}/results/results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to {json_path}")
        
        # Save summary as CSV
        summary_data = []
        for result in self.results:
            if result['success']:
                row = {
                    'experiment': result['experiment'],
                    'Ra': result['config']['Ra'],
                    'Pr': result['config']['Pr'],
                    'aspect_ratio': result['config']['aspect_ratio'],
                    'trad_time': result['traditional']['time'],
                    'pinn_time': result['pinn']['time'],
                    'speedup': result['traditional']['time'] / result['pinn']['time']
                }
                summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        csv_path = f"{self.output_dir}/results/summary_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"📊 Summary saved to {csv_path}")
    
    def generate_report(self):
        """Generate a markdown report of results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.output_dir}/results/report_{timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write("# Experiment Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"Total experiments: {len(self.results)}\n")
            successful = sum(1 for r in self.results if r['success'])
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {len(self.results) - successful}\n\n")
            
            f.write("## Results\n\n")
            
            for result in self.results:
                f.write(f"### {result['experiment']}\n\n")
                f.write(f"- **Status**: {'✅ Success' if result['success'] else '❌ Failed'}\n")
                
                if result['success']:
                    config = result['config']
                    f.write(f"- **Parameters**: Ra={config['Ra']:.1e}, ")
                    f.write(f"Pr={config['Pr']}, AR={config['aspect_ratio']}\n")
                    f.write(f"- **Traditional Time**: {result['traditional']['time']:.2f}s\n")
                    f.write(f"- **PINN Time**: {result['pinn']['time']:.2f}s\n")
                    speedup = result['traditional']['time'] / result['pinn']['time']
                    f.write(f"- **Speedup**: {speedup:.2f}x\n")
                
                f.write("\n")
        
        print(f"📄 Report saved to {report_path}")
    
    def plot_sweep_results(self, sweep_name, param_name, param_values):
        """Plot results from parameter sweep"""
        
        # Extract relevant results
        sweep_results = [r for r in self.results if sweep_name in r['experiment']]
        
        if not sweep_results:
            print("No results to plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Parameter Sweep: {sweep_name}', fontsize=16)
        
        # Extract data
        params = [r['config'][param_name] for r in sweep_results if r['success']]
        trad_times = [r['traditional']['time'] for r in sweep_results if r['success']]
        pinn_times = [r['pinn']['time'] for r in sweep_results if r['success']]
        
        # Plot 1: Computation times
        axes[0, 0].plot(params, trad_times, 'o-', label='Traditional', linewidth=2)
        axes[0, 0].plot(params, pinn_times, 's-', label='PINN', linewidth=2)
        axes[0, 0].set_xlabel(param_name)
        axes[0, 0].set_ylabel('Time (s)')
        axes[0, 0].set_title('Computation Time Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        if param_name == 'Ra':
            axes[0, 0].set_xscale('log')
        
        # Plot 2: Speedup
        speedup = [t/p for t, p in zip(trad_times, pinn_times)]
        axes[0, 1].plot(params, speedup, 'o-', color='green', linewidth=2)
        axes[0, 1].axhline(y=1.0, color='r', linestyle='--', label='Equal performance')
        axes[0, 1].set_xlabel(param_name)
        axes[0, 1].set_ylabel('Speedup (×)')
        axes[0, 1].set_title('PINN Speedup over Traditional')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        if param_name == 'Ra':
            axes[0, 1].set_xscale('log')
        
        # Plot 3: Errors (placeholder)
        axes[1, 0].set_title('Error Metrics (TODO)')
        axes[1, 0].set_xlabel(param_name)
        axes[1, 0].set_ylabel('Error')
        
        # Plot 4: Summary table
        axes[1, 1].axis('off')
        summary_text = f"Parameter Sweep Summary\n\n"
        summary_text += f"Parameter: {param_name}\n"
        summary_text += f"Values tested: {len(params)}\n"
        summary_text += f"Avg. traditional time: {np.mean(trad_times):.2f}s\n"
        summary_text += f"Avg. PINN time: {np.mean(pinn_times):.2f}s\n"
        summary_text += f"Avg. speedup: {np.mean(speedup):.2f}×\n"
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                       verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = f"{self.output_dir}/figures/{sweep_name}_{timestamp}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"📈 Figure saved to {fig_path}")
        
        plt.close()


def main():
    """Main entry point for batch experiments"""
    
    print("🌊 Convection Solver - Batch Experiment Runner")
    print("="*60)
    
    runner = ExperimentRunner()
    
    # Example usage
    print("\nAvailable experiments:")
    print("1. Run basic test suite")
    print("2. Run Rayleigh number sweep")
    print("3. Run Prandtl number sweep")
    print("4. Run aspect ratio sweep")
    print("5. Run all experiments")
    
    choice = input("\nSelect experiment (1-5): ")
    
    if choice == '1':
        test_cases = ['weak_convection', 'moderate_convection', 'air_ra1e4']
        runner.run_test_suite(test_cases)
    
    elif choice == '2':
        runner.run_parameter_sweep('Ra_sweep')
    
    elif choice == '3':
        runner.run_parameter_sweep('Pr_sweep')
    
    elif choice == '4':
        runner.run_parameter_sweep('AR_sweep')
    
    elif choice == '5':
        print("\n⚠️  This will run all experiments (may take hours!)")
        confirm = input("Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            runner.run_test_suite()
            runner.run_parameter_sweep('Ra_sweep')
            runner.run_parameter_sweep('Pr_sweep')
            runner.run_parameter_sweep('AR_sweep')
    
    print("\n✅ All experiments complete!")
    print(f"Results saved in: {runner.output_dir}/")


if __name__ == "__main__":
    main()
