# 🌊 AI-Powered Rayleigh-Bénard Convection Solver

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

An interactive web application that solves Rayleigh-Bénard convection problems using both **traditional numerical methods** and **Physics-Informed Neural Networks (PINNs)**. This project aims to create a generalist ML model for convection problems with robust validation against classical solvers.

## 🎯 Project Goals

1. **Develop a generalist ML model** capable of solving various convection problems across different parameter regimes
2. **Validate ML predictions** against traditional CFD solutions with comprehensive error metrics
3. **Enable real-time solving** for parameter exploration and design optimization
4. **Open-source contribution** to the fluid mechanics and ML research community
5. **Publication-ready** research with rigorous validation and novel insights

## ✨ Features

### Dual Solver Architecture
- **Traditional Finite Difference Solver**: Robust, well-validated numerical method using stream function-vorticity formulation
- **Physics-Informed Neural Network (PINN)**: Deep learning approach that embeds physical laws directly in the loss function

### Interactive Interface
- Real-time parameter adjustment (Rayleigh number, Prandtl number, aspect ratio)
- Side-by-side comparison of solver results
- Comprehensive error metrics (L2 norm, relative error, max error)
- Interactive visualization with Plotly

### Validation Framework
- Cross-validation between traditional and ML solutions
- Multiple field comparisons (temperature, velocity, stream function)
- Performance benchmarking (accuracy vs computational time)
- Training loss visualization

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/convection-solver.git
cd convection-solver

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📊 Usage

### 1. Set Problem Parameters
Use the sidebar to configure:
- **Rayleigh Number (Ra)**: 1e3 to 1e6 (controls convection strength)
- **Prandtl Number (Pr)**: 0.1 to 100 (fluid property)
- **Aspect Ratio**: 0.5 to 4.0 (domain geometry)
- **Grid Resolution**: 20×20 to 100×100 points

### 2. Configure Solvers
- **Traditional Solver**: Set max iterations and convergence tolerance
- **PINN**: Choose network architecture, training epochs, and learning rate

### 3. Run Simulations
- Click "Run Traditional Solver" to solve with finite differences
- Click "Run PINN Solver" to train and predict with neural network
- Compare results side-by-side

### 4. Analyze Results
- View temperature, velocity, and stream function fields
- Examine error metrics and convergence
- Compare computational performance

## 🔬 Technical Details

### Governing Equations

The Rayleigh-Bénard convection is governed by the Boussinesq approximation:

**Continuity:**
```
∇·u = 0
```

**Momentum (with buoyancy):**
```
∂u/∂t + (u·∇)u = -∇p + Pr·∇²u + Ra·Pr·T·ĵ
```

**Energy:**
```
∂T/∂t + u·∇T = ∇²T
```

### Traditional Solver

- **Method**: Finite difference with stream function-vorticity formulation
- **Discretization**: Central differences for spatial derivatives
- **Time Integration**: Euler explicit scheme
- **Boundary Conditions**: 
  - Hot bottom wall (T=1)
  - Cold top wall (T=0)
  - No-slip walls (u=v=0)
  - Adiabatic side walls

### PINN Architecture

- **Input**: Spatial coordinates (x, y)
- **Output**: Temperature (T), stream function (ψ), velocities (u, v)
- **Architecture**: Fully connected neural network with tanh activation
- **Loss Function**: 
  ```
  L = L_physics + λ·L_boundary
  ```
  where:
  - L_physics: PDE residuals in domain
  - L_boundary: Boundary condition enforcement
  - λ: Boundary loss weight (typically 10)

## 📈 Validation Metrics

The application computes several metrics to validate ML predictions:

1. **L2 Error**: `√(Σ(y_true - y_pred)²/n)`
2. **Relative L2 Error**: `L2_error / √(Σ(y_true)²/n)`
3. **Maximum Absolute Error**: `max(|y_true - y_pred|)`
4. **Computational Time Comparison**

## 🎓 Research Applications

This tool is designed to support research in:

- **Physics-Informed Machine Learning**: Novel PINN architectures and training strategies
- **Computational Fluid Dynamics**: Fast parameter sweeps and design optimization
- **Heat Transfer**: Natural convection analysis and Nusselt number predictions
- **Model Validation**: Benchmark for new ML methods in fluid mechanics

## 🛣️ Roadmap

### Phase 1: Core Functionality ✅
- [x] Traditional finite difference solver
- [x] Basic PINN implementation
- [x] Interactive Streamlit interface
- [x] Side-by-side comparison

### Phase 2: Enhanced Features 🚧
- [ ] Multiple convection types (natural, mixed, forced)
- [ ] Turbulent regime handling (high Ra numbers)
- [ ] 3D convection problems
- [ ] Adaptive mesh refinement
- [ ] Advanced PINN architectures (Fourier Features, Multi-scale)

### Phase 3: Advanced Capabilities 📋
- [ ] Transfer learning across parameter regimes
- [ ] Uncertainty quantification
- [ ] Experimental data integration
- [ ] Real-time optimization
- [ ] Multi-physics coupling

### Phase 4: Publication & Deployment 📋
- [ ] Comprehensive validation study
- [ ] Performance benchmarking
- [ ] Manuscript preparation
- [ ] Cloud deployment
- [ ] API for programmatic access

## 📚 Citation

If you use this software in your research, please cite:

```bibtex
@software{convection_solver_2024,
  title={AI-Powered Rayleigh-Bénard Convection Solver},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/convection-solver}
}
```

## 🤝 Contributing

Contributions are welcome! Areas where you can help:

- **Algorithm improvements**: Better solvers, optimization strategies
- **New features**: Additional convection problems, visualization options
- **Documentation**: Tutorials, examples, theoretical background
- **Testing**: Validation cases, bug reports
- **Performance**: Code optimization, GPU acceleration

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Classical solver algorithms based on well-established CFD literature
- PINN implementation inspired by Raissi et al. (2019)
- Built with Streamlit, PyTorch, NumPy, SciPy, and Plotly
- Community feedback and contributions

## 📞 Contact

For questions, suggestions, or collaboration:
- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/convection-solver/issues)
- **Email**: your.email@example.com
- **Twitter**: @yourusername

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Built with ❤️ for the fluid mechanics and machine learning community**
