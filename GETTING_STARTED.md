# Getting Started with Convection Solver

Welcome! This guide will help you get up and running with the AI-Powered Rayleigh-Bénard Convection Solver in just a few minutes.

## 📋 Prerequisites

- **Python**: Version 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Storage**: ~500MB for installation
- **GPU**: Optional but recommended for faster PINN training

## 🚀 Installation

### Method 1: Automated Setup (Recommended)

```bash
# Navigate to project directory
cd convection_solver

# Run setup script (Linux/macOS)
./setup.sh

# Or manually for Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Method 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import streamlit; import torch; print('✓ Installation successful!')"
```

## 🎮 Running the Application

### Start the Streamlit App

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### First Run Tutorial

#### Step 1: Set Basic Parameters (Left Sidebar)

Start with these recommended settings:
- **Rayleigh Number**: 10000 (1e4)
- **Prandtl Number**: 0.71 (air)
- **Aspect Ratio**: 2.0
- **Grid Points (X)**: 50
- **Grid Points (Y)**: 50

#### Step 2: Configure Solvers

**Traditional Solver:**
- **Max Iterations**: 1000
- **Convergence Tolerance**: 1e-5

**Neural Network:**
- **Hidden Layers**: 4
- **Neurons per Layer**: 50
- **Training Epochs**: 5000
- **Learning Rate**: 0.001

#### Step 3: Run Traditional Solver

1. Click the "🧮 Run Traditional Solver" button
2. Wait for convergence (watch the progress bar)
3. Note the computation time displayed

**Expected Result:** 
- Converges in ~500-1000 iterations
- Takes ~30-90 seconds
- You'll see a success message

#### Step 4: Run PINN Solver

1. Click the "🧠 Run PINN Solver" button
2. Monitor training progress
3. Wait for completion

**Expected Result:**
- Trains for 5000 epochs
- Takes ~2-5 minutes
- Final loss should be < 1e-3

#### Step 5: Compare Results

After both solvers complete:

1. **View Side-by-Side Comparison**
   - Select "Temperature" from dropdown
   - Examine the three panels:
     - Traditional solution
     - PINN solution
     - Absolute difference

2. **Check Error Metrics**
   - L2 Error (should be < 0.1)
   - Relative L2 Error (should be < 10%)
   - Max Error (should be < 0.5)

3. **Analyze Performance**
   - Compare computation times
   - Note the speedup factor

4. **Explore Other Fields**
   - Try "Stream Function"
   - Try "U Velocity"
   - Try "V Velocity"

## 📊 Understanding the Results

### Temperature Field
- **Hot bottom** (red): Temperature = 1.0
- **Cold top** (blue): Temperature = 0.0
- **Convection cells**: Visible circulation patterns
- **Symmetry**: Should be approximately symmetric

### Stream Function
- Shows circulation patterns
- Closed contours indicate convection cells
- Higher values = stronger circulation

### Velocity Fields
- **U Velocity**: Horizontal motion
- **V Velocity**: Vertical motion
- Combined: Shows flow direction and magnitude

### Error Metrics
- **L2 Error**: Overall difference between solutions
- **Relative L2 Error**: Percentage-based error
- **Max Error**: Largest point-wise difference

## 🔬 Running Your First Experiment

Let's run a simple parameter study:

### Experiment: Effect of Rayleigh Number

1. **Run Case 1: Ra = 1e3** (weak convection)
   - Set Ra = 1000
   - Run both solvers
   - Save/screenshot results

2. **Run Case 2: Ra = 1e4** (moderate)
   - Set Ra = 10000
   - Run both solvers
   - Compare with Case 1

3. **Run Case 3: Ra = 1e5** (strong)
   - Set Ra = 100000
   - Increase grid to 80×80
   - Increase PINN epochs to 8000
   - Run both solvers
   - Compare with previous cases

### What to Observe:
- **Increasing convection strength** with higher Ra
- **More complex patterns** at higher Ra
- **Longer computation times** for complex cases
- **PINN generalization** across different Ra values

## 🎯 Tips for Success

### For Traditional Solver:
1. **Grid Resolution**: Higher Ra needs finer grid
2. **Iterations**: Increase if not converging
3. **Tolerance**: Can relax to 1e-4 for faster results
4. **Aspect Ratio**: 2.0 is standard, try 1.0-4.0

### For PINN Solver:
1. **Start Small**: Begin with 2000-3000 epochs
2. **Monitor Loss**: Should decrease steadily
3. **Architecture**: 4-6 layers usually sufficient
4. **Learning Rate**: 1e-3 is good starting point
5. **Patience**: Training takes time but results improve

### For Both:
1. **Validate**: Always cross-check solutions
2. **Document**: Note parameter choices
3. **Iterate**: Adjust based on results
4. **Save**: Keep track of successful configurations

## ⚠️ Common Issues and Solutions

### Issue 1: PINN Not Converging
**Symptoms:** Loss stays high, poor agreement with traditional solver

**Solutions:**
- Increase training epochs (try 10000+)
- Add more layers or neurons
- Decrease learning rate to 5e-4
- Check that traditional solver converged first

### Issue 2: Traditional Solver Slow
**Symptoms:** Taking very long to converge

**Solutions:**
- Reduce grid resolution (try 30×30)
- Increase tolerance to 1e-4
- Reduce max iterations
- Start with lower Ra number

### Issue 3: Large Errors Between Methods
**Symptoms:** L2 error > 0.5, visual differences obvious

**Solutions:**
- Both methods may need higher resolution
- PINN may need more training
- Check boundary conditions are correct
- Verify traditional solver converged

### Issue 4: Out of Memory
**Symptoms:** Python crashes, memory errors

**Solutions:**
- Reduce grid size
- Use smaller PINN architecture
- Close other applications
- Reduce batch size (in code)

## 📚 Next Steps

### Learning Resources:
1. **QUICKSTART.md** - Detailed parameter guide
2. **README.md** - Full project documentation
3. **configs.py** - Predefined test cases
4. **PROJECT_STRUCTURE.md** - Code organization

### Advanced Features:
1. **Batch Experiments** - Run systematic studies
   ```bash
   python batch_experiments.py
   ```

2. **Custom Test Cases** - Define in configs.py
3. **Analysis Tab** - View training history
4. **Parameter Sweeps** - Explore Ra, Pr, AR space

### Research Applications:
1. Validate PINN across parameter ranges
2. Study convection onset and transitions
3. Optimize network architectures
4. Compare with experimental data
5. Develop transfer learning strategies

## 🤝 Getting Help

### Resources:
- **Documentation**: Check README.md and other .md files
- **GitHub Issues**: Report bugs or ask questions
- **Examples**: See configs.py for test cases
- **Community**: Join discussions (if available)

### Debugging Checklist:
- [ ] Python version ≥ 3.8
- [ ] All packages installed
- [ ] Virtual environment activated
- [ ] Sufficient RAM available
- [ ] Traditional solver converges
- [ ] PINN loss decreasing
- [ ] Boundary conditions correct

## 🎉 You're Ready!

You now have everything needed to:
- ✅ Run convection simulations
- ✅ Compare traditional vs ML methods
- ✅ Validate PINN predictions
- ✅ Conduct parameter studies
- ✅ Generate publication-quality results

**Happy solving! 🌊**

---

*For more information, see the full README.md or visit the project repository.*
