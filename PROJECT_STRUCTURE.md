# Project Structure

```
convection_solver/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── CONTRIBUTING.md             # Contribution guidelines
├── QUICKSTART.md              # Quick start guide
├── configs.py                 # Configuration and test cases
├── batch_experiments.py       # Batch experiment runner
│
├── docs/                      # Documentation (future)
│   ├── theory.md             # Theoretical background
│   ├── api.md                # API documentation
│   └── tutorials/            # Tutorial notebooks
│
├── tests/                     # Unit tests (future)
│   ├── test_traditional_solver.py
│   ├── test_pinn_solver.py
│   └── test_validation.py
│
├── experiments/               # Experiment outputs
│   ├── results/              # JSON and CSV results
│   ├── figures/              # Generated plots
│   └── data/                 # Saved model weights
│
├── notebooks/                 # Jupyter notebooks (future)
│   ├── 01_introduction.ipynb
│   ├── 02_parameter_study.ipynb
│   └── 03_validation.ipynb
│
└── utils/                     # Utility functions (future)
    ├── visualization.py      # Plotting helpers
    ├── metrics.py           # Error metrics
    └── data_generation.py   # Training data generation
```

## File Descriptions

### Core Files

**app.py**
- Main Streamlit application
- Contains both TraditionalSolver and PINNSolver classes
- Interactive UI for running experiments
- Visualization and comparison tools

**configs.py**
- Predefined test cases and configurations
- Parameter sweep definitions
- Architecture experiments
- Training tips and best practices

**batch_experiments.py**
- Automated experiment runner
- Systematic parameter sweeps
- Result logging and analysis
- Figure generation

### Documentation

**README.md**
- Project overview and features
- Installation instructions
- Usage guide
- Roadmap and goals

**QUICKSTART.md**
- Step-by-step getting started guide
- Recommended settings for different use cases
- Common issues and solutions
- Tips for successful experiments

**CONTRIBUTING.md**
- Guidelines for contributors
- Code style and standards
- Pull request process
- Development setup

### Future Additions

**tests/**
- Unit tests for solver validation
- Integration tests
- Regression tests
- Continuous integration setup

**docs/**
- Detailed theoretical documentation
- Mathematical formulations
- API reference
- Advanced tutorials

**notebooks/**
- Interactive tutorials
- Parameter studies
- Validation analyses
- Publication-ready figures

**utils/**
- Reusable utility functions
- Visualization helpers
- Data processing tools
- Performance profiling

## Key Components

### TraditionalSolver Class
```python
class TraditionalSolver:
    - Finite difference implementation
    - Stream function-vorticity formulation
    - Poisson solver
    - Boundary condition handling
    - Velocity computation
```

### PINNSolver Class
```python
class PINNSolver:
    - Physics-informed neural network
    - Automatic differentiation for PDEs
    - Training loop with physics loss
    - Boundary condition enforcement
    - Grid-based prediction
```

### ExperimentRunner Class
```python
class ExperimentRunner:
    - Batch experiment execution
    - Result logging (JSON/CSV)
    - Automatic report generation
    - Parameter sweep visualization
```

## Data Flow

```
User Input (Streamlit UI)
    ↓
Parameter Configuration
    ↓
Solver Selection
    ↓
┌─────────────────┬──────────────────┐
│ Traditional     │ PINN             │
│ Solver          │ Solver           │
└─────────────────┴──────────────────┘
    ↓                    ↓
Results Collection
    ↓
Validation & Comparison
    ↓
Visualization (Plotly)
    ↓
Display to User
```

## Development Workflow

1. **Local Development**
   ```bash
   streamlit run app.py
   ```

2. **Run Tests**
   ```bash
   pytest tests/
   ```

3. **Run Experiments**
   ```bash
   python batch_experiments.py
   ```

4. **Format Code**
   ```bash
   black *.py
   flake8 *.py
   ```

## Deployment Options

### Local
- Run with Streamlit locally
- No external dependencies needed

### Streamlit Cloud
- Deploy directly from GitHub
- Free hosting for public repos
- Automatic updates on push

### Docker
- Containerized deployment
- Consistent environment
- Easy scaling

### Cloud Platforms
- AWS, GCP, Azure
- Scalable compute resources
- API endpoints

## Extension Points

### Adding New Solvers
1. Create new solver class
2. Implement solve() method
3. Return standardized results dictionary
4. Add to UI solver selection

### Adding New Visualizations
1. Add plotting function
2. Create Plotly figure
3. Add to visualization tab
4. Update documentation

### Adding New Test Cases
1. Define in configs.py
2. Add to TEST_CASES dictionary
3. Document parameters
4. Add validation targets if known

## Performance Considerations

### Traditional Solver
- O(n²) for grid points
- Memory: ~100MB for 100×100 grid
- Time: Linear with iterations

### PINN Solver
- O(layers × neurons) parameters
- GPU acceleration recommended
- Memory: ~500MB typical
- Time: Linear with epochs

## Best Practices

1. **Always validate** ML predictions against traditional solver
2. **Start simple** - low Ra, coarse grid, then scale up
3. **Version control** experiment configurations
4. **Document** parameter choices and results
5. **Use batch experiments** for systematic studies
6. **Monitor** training convergence closely
7. **Save** intermediate results frequently
8. **Visualize** errors and residuals
9. **Compare** computational costs
10. **Share** findings with community

## Future Enhancements

- [ ] Modular architecture (separate files for solvers)
- [ ] Database for experiment tracking
- [ ] Web API for programmatic access
- [ ] Real-time collaboration features
- [ ] Integration with experiment management tools
- [ ] Advanced visualization (3D, animations)
- [ ] Automated hyperparameter tuning
- [ ] Model compression and deployment
