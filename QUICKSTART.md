# Quick Start Guide

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

## First Run - Recommended Settings

### For Quick Testing (Low Resolution)
- **Rayleigh Number**: 1e4
- **Prandtl Number**: 0.71 (air)
- **Grid Points**: 30×30
- **Traditional Solver**: 500 iterations
- **PINN**: 2000 epochs, 4 hidden layers, 50 neurons

**Expected time**: ~1-2 minutes total

### For Better Accuracy (Medium Resolution)
- **Rayleigh Number**: 1e5
- **Prandtl Number**: 0.71
- **Grid Points**: 50×50
- **Traditional Solver**: 1000 iterations
- **PINN**: 5000 epochs, 4 hidden layers, 50 neurons

**Expected time**: ~5-10 minutes total

### For Publication Quality (High Resolution)
- **Rayleigh Number**: Variable (1e4 - 1e6)
- **Prandtl Number**: 0.71
- **Grid Points**: 100×100
- **Traditional Solver**: 2000 iterations
- **PINN**: 10000 epochs, 6 hidden layers, 100 neurons

**Expected time**: ~30-60 minutes total

## Step-by-Step Workflow

### Step 1: Configure Parameters
1. Open the sidebar
2. Set Rayleigh number (start with 1e4)
3. Set Prandtl number (0.71 for air)
4. Choose aspect ratio (2.0 is standard)

### Step 2: Run Traditional Solver
1. Click "🧮 Run Traditional Solver"
2. Wait for convergence (watch progress bar)
3. Note the computation time

### Step 3: Run PINN Solver
1. Click "🧠 Run PINN Solver"
2. Monitor training progress
3. Wait for completion

### Step 4: Compare Results
1. Select field to compare (Temperature recommended first)
2. Examine visual differences
3. Check error metrics below
4. Try different fields (Stream Function, Velocities)

### Step 5: Analyze
1. Switch to "Analysis" tab
2. View PINN training loss curve
3. Check convergence behavior

## Common Issues & Solutions

### Issue: PINN not converging well
**Solution**: 
- Increase training epochs
- Add more hidden layers
- Reduce learning rate
- Increase boundary loss weight in code

### Issue: Traditional solver too slow
**Solution**:
- Reduce grid resolution
- Decrease max iterations
- Increase convergence tolerance

### Issue: Large errors between methods
**Solution**:
- Both methods may need more resolution/training
- Check if traditional solver converged
- Try lower Rayleigh number first

## Parameter Recommendations by Rayleigh Number

### Ra = 1e3 - 1e4 (Weak Convection)
- Grid: 30×30 minimum
- PINN epochs: 2000-5000
- Easy to solve, good for testing

### Ra = 1e4 - 1e5 (Moderate Convection)
- Grid: 50×50 recommended
- PINN epochs: 5000-8000
- Standard test case

### Ra = 1e5 - 1e6 (Strong Convection)
- Grid: 80×80 or higher
- PINN epochs: 8000-15000
- Challenging, may need careful tuning

### Ra > 1e6 (Turbulent Regime)
- Grid: 100×100 minimum
- PINN epochs: 15000+
- Very challenging, consider advanced methods

## Tips for Best Results

1. **Always run traditional solver first** - establishes ground truth
2. **Start with low Ra numbers** - easier to validate
3. **Gradually increase complexity** - build confidence in models
4. **Compare multiple fields** - comprehensive validation
5. **Save your configurations** - document successful parameters
6. **Check convergence** - don't trust unconverged solutions
7. **Use appropriate resolution** - higher Ra needs finer grids

## Next Steps

After getting familiar with the basic interface:

1. **Experiment with parameters** - explore Ra-Pr space
2. **Compare computation times** - analyze ML speedup
3. **Study error patterns** - understand model limitations
4. **Try different architectures** - optimize PINN design
5. **Document findings** - prepare for publication

## Getting Help

- Check the "About" tab for detailed explanations
- Review error metrics to diagnose issues
- Refer to README.md for technical details
- Open GitHub issues for bugs or questions

Happy solving! 🌊
