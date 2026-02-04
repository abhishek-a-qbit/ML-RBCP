"""
Systematic Rayleigh-Bénard Convection Solver
Complete problem setup, visualization, and solution comparison
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import seaborn as sns  # Commented out for Streamlit Cloud compatibility
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os

# Add the current directory to Python path to import our solvers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import solvers, but handle gracefully for Streamlit Cloud
try:
    from solvers import SolverFactory, compute_nusselt_number, validate_solution
    SOLVERS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Advanced solvers not available: {e}")
    SOLVERS_AVAILABLE = False
    
    # Create dummy solver factory for Streamlit Cloud
    class DummySolverFactory:
        @staticmethod
        def get_available_methods():
            return {
                'finite_difference': {
                    'name': 'Finite Difference Method',
                    'description': 'Traditional CFD approach using grid discretization',
                    'pros': ['Robust', 'Widely applicable', 'Easy to implement'],
                    'cons': ['Second-order accuracy', 'Numerical diffusion']
                }
            }
        
        @staticmethod
        def create_solver(method, **kwargs):
            return None
    
    SolverFactory = DummySolverFactory

# Set page config
st.set_page_config(
    page_title="Systematic Rayleigh-Bénard Convection Analysis", 
    layout="wide", 
    page_icon="🌊"
)

# ============================================================================
# PROBLEM VISUALIZATION FUNCTIONS
# ============================================================================

def draw_problem_setup():
    """Draw the Rayleigh-Bénard convection problem setup"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Physical setup
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.set_title('Rayleigh-Bénard Convection Problem Setup', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x (Width)', fontsize=12)
    ax1.set_ylabel('y (Height)', fontsize=12)
    
    # Draw domain
    domain = patches.Rectangle((0, 0), 2, 1, linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.3)
    ax1.add_patch(domain)
    
    # Hot bottom boundary
    bottom = patches.Rectangle((0, 0), 2, 0.05, linewidth=0, facecolor='red', alpha=0.8)
    ax1.add_patch(bottom)
    ax1.text(1, -0.1, 'HOT WALL (T = T_h)', ha='center', fontsize=11, color='red', fontweight='bold')
    
    # Cold top boundary  
    top = patches.Rectangle((0, 0.95), 2, 0.05, linewidth=0, facecolor='blue', alpha=0.8)
    ax1.add_patch(top)
    ax1.text(1, 1.1, 'COLD WALL (T = T_c)', ha='center', fontsize=11, color='blue', fontweight='bold')
    
    # Adiabatic side walls
    left = patches.Rectangle((0, 0), 0.05, 1, linewidth=0, facecolor='gray', alpha=0.6)
    ax1.add_patch(left)
    right = patches.Rectangle((1.95, 0), 0.05, 1, linewidth=0, facecolor='gray', alpha=0.6)
    ax1.add_patch(right)
    ax1.text(-0.2, 0.5, 'ADIABATIC', ha='center', fontsize=10, rotation=90, va='center')
    ax1.text(2.2, 0.5, 'ADIABATIC', ha='center', fontsize=10, rotation=90, va='center')
    
    # Add convection cells
    x = np.linspace(0.1, 1.9, 100)
    y = np.linspace(0.1, 0.9, 50)
    X, Y = np.meshgrid(x, y)
    
    # Create convection pattern
    U = np.sin(2*np.pi*X/2) * np.cos(np.pi*Y)
    V = np.cos(2*np.pi*X/2) * np.sin(np.pi*Y)
    
    # Draw streamlines
    stream = ax1.streamplot(X, Y, U, V, color='darkblue', density=1.5, linewidth=1)
    # Set alpha for the streamlines
    stream.lines.set_alpha(0.7)
    
    # Add dimension arrows
    ax1.annotate('', xy=(2, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='<->', lw=2))
    ax1.text(1, -0.05, 'L', ha='center', fontsize=12)
    ax1.annotate('', xy=(0, 1), xytext=(0, 0), arrowprops=dict(arrowstyle='<->', lw=2))
    ax1.text(-0.1, 0.5, 'H', ha='center', fontsize=12)
    
    # Right: Parameters diagram
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Physical Parameters', fontsize=14, fontweight='bold')
    
    # Create parameter boxes
    params_text = [
        ('Temperature Field', 'T(x,y,t)', 'Primary variable'),
        ('Velocity Field', 'u(x,y,t), v(x,y,t)', 'Fluid motion'),
        ('Pressure Field', 'p(x,y,t)', 'Thermodynamic pressure'),
        ('Density', 'ρ(x,y,t)', 'Fluid density'),
        ('Viscosity', 'μ', 'Dynamic viscosity'),
        ('Thermal Conductivity', 'k', 'Heat transfer'),
        ('Thermal Expansion', 'α', 'Buoyancy coefficient'),
        ('Specific Heat', 'c_p', 'Heat capacity'),
        ('Gravity', 'g', 'Body force')
    ]
    
    y_pos = 0.9
    for param, symbol, desc in params_text:
        ax2.text(0.1, y_pos, f'• {param}', fontsize=11, fontweight='bold')
        ax2.text(0.4, y_pos, f'{symbol}', fontsize=11, style='italic')
        ax2.text(0.7, y_pos, f'{desc}', fontsize=10, style='italic')
        y_pos -= 0.1
    
    plt.tight_layout()
    return fig

def display_governing_equations():
    """Display complete governing equations"""
    st.markdown("""
    ## 🔬 Governing Equations
    
    The Rayleigh-Bénard convection problem is governed by the **Boussinesq approximation** of the Navier-Stokes equations:
    
    ### **1. Continuity Equation (Mass Conservation)**
    $$\\frac{\\partial \\rho}{\\partial t} + \\nabla \\cdot (\\rho \\mathbf{u}) = 0$$
    
    For incompressible flow: $$\\nabla \\cdot \\mathbf{u} = 0$$
    
    ### **2. Momentum Equations (Navier-Stokes)**
    $$\\frac{\\partial \\mathbf{u}}{\\partial t} + (\\mathbf{u} \\cdot \\nabla)\\mathbf{u} = -\\frac{1}{\\rho_0}\\nabla p + \\nu \\nabla^2 \\mathbf{u} + \\mathbf{g}$$
    
    With Boussinesq approximation for buoyancy:
    $$\\frac{\\partial u}{\\partial t} + (\\mathbf{u} \\cdot \\nabla)u = -\\frac{1}{\\rho_0}\\frac{\\partial p}{\\partial x} + \\nu \\nabla^2 u$$
    $$\\frac{\\partial v}{\\partial t} + (\\mathbf{u} \\cdot \\nabla)v = -\\frac{1}{\\rho_0}\\frac{\\partial p}{\\partial y} + \\nu \\nabla^2 v + g\\alpha(T-T_0)$$
    
    ### **3. Energy Equation (Heat Transfer)**
    $$\\frac{\\partial T}{\\partial t} + (\\mathbf{u} \\cdot \\nabla)T = \\kappa \\nabla^2 T$$
    
    Where:
    - $\\mathbf{u} = (u, v)$ is the velocity vector
    - $p$ is the pressure
    - $T$ is the temperature
    - $\\rho_0$ is the reference density
    - $\\nu = \\mu/\\rho_0$ is the kinematic viscosity
    - $\\kappa = k/(\\rho_0 c_p)$ is the thermal diffusivity
    - $g$ is gravitational acceleration
    - $\\alpha$ is the thermal expansion coefficient
    
    ### **4. Dimensionless Parameters**
    
    **Rayleigh Number (Ra):**
    $$Ra = \\frac{g \\alpha \\Delta T L^3}{\\nu \\kappa}$$
    - Ratio of buoyancy to viscous forces
    - $Ra < 1708$: Pure conduction
    - $Ra > 1708$: Onset of convection
    
    **Prandtl Number (Pr):**
    $$Pr = \\frac{\\nu}{\\kappa} = \\frac{\\mu c_p}{k}$$
    - Ratio of momentum to thermal diffusivity
    - Air: $Pr \\approx 0.71$, Water: $Pr \\approx 7.0$
    
    **Nusselt Number (Nu):**
    $$Nu = \\frac{Q_{actual}}{Q_{conduction}} = -\\frac{L}{\\Delta T}\\frac{\\partial T}{\\partial y}\\Big|_{wall}$$
    - Ratio of actual to conductive heat transfer
    - $Nu = 1$: Pure conduction
    - $Nu > 1$: Convective enhancement
    
    ### **5. Boundary Conditions**
    
    **Temperature:**
    - Bottom wall ($y=0$): $T = T_h$ (hot)
    - Top wall ($y=H$): $T = T_c$ (cold)
    - Side walls ($x=0, L$): $\\frac{\\partial T}{\\partial x} = 0$ (adiabatic)
    
    **Velocity (No-slip):**
    - All walls: $u = v = 0$
    
    **Pressure:**
    - Reference pressure at one point
    """)

def create_solution_comparison(results1, results2, method1, method2):
    """Create comprehensive solution comparison"""
    if results1 is None or results2 is None:
        return None
    
    # Check grid compatibility and interpolate if necessary
    x1, y1 = results1['x'], results1['y']
    x2, y2 = results2['x'], results2['y']
    
    # If grids are different, interpolate to the finer grid
    if len(x1) != len(x2) or len(y1) != len(y2):
        # Use the finer grid for comparison
        if len(x1) * len(y1) >= len(x2) * len(y2):
            # Interpolate results2 to results1's grid
            from scipy.interpolate import interp2d
            X1, Y1 = np.meshgrid(x1, y1)
            X2, Y2 = np.meshgrid(x2, y2)
            
            results2_interp = {}
            for field in ['temperature', 'stream_function', 'u_velocity', 'v_velocity']:
                if field in results2 and field in results1:
                    f = interp2d(x2, y2, results2[field], kind='linear')
                    results2_interp[field] = f(x1, y1)
            
            # Use interpolated results2 for comparison
            temp_results2 = results2.copy()
            for field, data in results2_interp.items():
                temp_results2[field] = data
            results2 = temp_results2
        else:
            # Interpolate results1 to results2's grid
            from scipy.interpolate import interp2d
            X1, Y1 = np.meshgrid(x1, y1)
            X2, Y2 = np.meshgrid(x2, y2)
            
            results1_interp = {}
            for field in ['temperature', 'stream_function', 'u_velocity', 'v_velocity']:
                if field in results1 and field in results2:
                    f = interp2d(x1, y1, results1[field], kind='linear')
                    results1_interp[field] = f(x2, y2)
            
            # Use interpolated results1 for comparison
            temp_results1 = results1.copy()
            for field, data in results1_interp.items():
                temp_results1[field] = data
            results1 = temp_results1
    
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            f'{method1.replace("_", " ").title()} - Temperature',
            f'{method2.replace("_", " ").title()} - Temperature',
            'Temperature Difference',
            f'{method1.replace("_", " ").title()} - Stream Function',
            f'{method2.replace("_", " ").title()} - Stream Function',
            'Stream Function Difference',
            f'{method1.replace("_", " ").title()} - Velocity Magnitude',
            f'{method2.replace("_", " ").title()} - Velocity Magnitude',
            'Velocity Magnitude Difference'
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.08
    )
    
    fields = ['temperature', 'stream_function', 'u_velocity']
    field_names = ['Temperature', 'Stream Function', 'Velocity Magnitude']
    
    for i, (field, field_name) in enumerate(zip(fields, field_names)):
        # Get data
        if field in results1 and field in results2:
            data1 = results1[field]
            data2 = results2[field]
            
            # Handle velocity magnitude
            if field == 'u_velocity':
                data1 = np.sqrt(results1['u_velocity']**2 + results1['v_velocity']**2)
                data2 = np.sqrt(results2['u_velocity']**2 + results2['v_velocity']**2)
            
            # Check for NaN or invalid data
            if np.any(np.isnan(data1)) or np.any(np.isnan(data2)):
                # Create placeholder plots for invalid data
                for col in range(1, 4):
                    fig.add_trace(
                        go.Heatmap(z=np.zeros((10, 10)), x=[0, 1], y=[0, 1],
                                  colorscale='Gray', showscale=False),
                        row=i+1, col=col
                    )
                continue
            
            # Determine color scale
            vmin = min(np.min(data1), np.min(data2))
            vmax = max(np.max(data1), np.max(data2))
            
            # Plot method 1
            fig.add_trace(
                go.Heatmap(z=data1, x=results1['x'], y=results1['y'],
                          colorscale='RdBu_r' if field_name != 'Stream Function' else 'Viridis',
                          zmin=vmin, zmax=vmax, showscale=False),
                row=i+1, col=1
            )
            
            # Plot method 2
            fig.add_trace(
                go.Heatmap(z=data2, x=results2['x'], y=results2['y'],
                          colorscale='RdBu_r' if field_name != 'Stream Function' else 'Viridis',
                          zmin=vmin, zmax=vmax, showscale=False),
                row=i+1, col=2
            )
            
            # Plot difference
            diff = np.abs(data1 - data2)
            fig.add_trace(
                go.Heatmap(z=diff, x=results1['x'], y=results1['y'],
                          colorscale='Hot', showscale=(i==2),
                          colorbar=dict(title="|Difference|", x=1.15) if i==2 else None),
                row=i+1, col=3
            )
    
    fig.update_layout(
        height=1200,
        title_text="Comprehensive Solution Comparison",
        showlegend=False
    )
    
    # Update axis labels
    for i in range(1, 4):
        fig.update_xaxes(title_text="x", row=i, col=2)
        fig.update_yaxes(title_text="y", row=i, col=1)
    
    return fig

def create_parameter_sensitivity_plot():
    """Create parameter sensitivity analysis plot"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Rayleigh number effect
    Ra_range = np.logspace(3, 6, 50)
    Nu_critical = 1 + 0.5 * ((Ra_range/1708) - 1)**0.5  # Simplified correlation
    ax1.loglog(Ra_range, Nu_critical, 'b-', linewidth=2)
    ax1.axhline(y=1, color='r', linestyle='--', label='Pure conduction')
    ax1.axvline(x=1708, color='g', linestyle='--', label='Critical Ra')
    ax1.set_xlabel('Rayleigh Number (Ra)')
    ax1.set_ylabel('Nusselt Number (Nu)')
    ax1.set_title('Heat Transfer vs Rayleigh Number')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Prandtl number effect
    Pr_range = np.logspace(-2, 2, 50)
    stability_factor = np.sqrt(Pr_range / (1 + Pr_range))
    ax2.loglog(Pr_range, stability_factor, 'r-', linewidth=2)
    ax2.axvline(x=0.71, color='b', linestyle='--', label='Air')
    ax2.axvline(x=7.0, color='g', linestyle='--', label='Water')
    ax2.set_xlabel('Prandtl Number (Pr)')
    ax2.set_ylabel('Stability Factor')
    ax2.set_title('Flow Stability vs Prandtl Number')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Flow regimes
    Ra_regimes = [1e3, 1e4, 1e5, 1e6, 1e7]
    regime_names = ['Conduction', 'Steady Convection', 'Oscillatory', 'Chaotic', 'Turbulent']
    colors = ['blue', 'green', 'orange', 'red', 'darkred']
    
    ax3.bar(range(len(Ra_regimes)), Ra_regimes, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(Ra_regimes)))
    ax3.set_xticklabels(regime_names, rotation=45, ha='right')
    ax3.set_ylabel('Rayleigh Number')
    ax3.set_title('Flow Regimes')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Parameter space
    Ra_grid, Pr_grid = np.meshgrid(np.logspace(3, 7, 50), np.logspace(-1, 2, 50))
    stability = Ra_grid * Pr_grid / 1708
    
    contour = ax4.contourf(Ra_grid, Pr_grid, stability, levels=20, cmap='RdYlBu_r')
    ax4.contour(Ra_grid, Pr_grid, stability, levels=[1], colors='black', linewidths=2)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('Rayleigh Number (Ra)')
    ax4.set_ylabel('Prandtl Number (Pr)')
    ax4.set_title('Stability Parameter (Ra·Pr/1708)')
    plt.colorbar(contour, ax=ax4, label='Stability')
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Title and description
st.title("🌊 Systematic Rayleigh-Bénard Convection Analysis")
st.markdown("""
Complete computational analysis of Rayleigh-Bénard convection with systematic problem setup, 
governing equations, and comprehensive solution comparison.
""")

# Create main tabs
main_tabs = st.tabs(["📋 Problem Setup", "🔬 Solution Methods", "📊 Results Analysis", "� Method Comparison", "� Theory"])

with main_tabs[0]:
    st.header("📋 Problem Setup & Parameters")
    
    # Problem visualization
    st.subheader("Physical Problem Configuration")
    problem_fig = draw_problem_setup()
    st.pyplot(problem_fig)
    
    # Governing equations
    st.markdown("---")
    display_governing_equations()
    
    # Parameter space analysis
    st.markdown("---")
    st.subheader("Parameter Space Analysis")
    param_fig = create_parameter_sensitivity_plot()
    st.pyplot(param_fig)
    
    # Interactive parameter selection
    st.markdown("---")
    st.subheader("Simulation Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Physical Parameters**")
        Ra = st.number_input(
            "Rayleigh Number (Ra)", 
            min_value=1e3, 
            max_value=1e7, 
            value=1e4, 
            format="%.0e",
            help="Controls convection strength"
        )
        
        Pr = st.number_input(
            "Prandtl Number (Pr)", 
            min_value=0.01, 
            max_value=100.0, 
            value=0.71,
            help="Fluid property ratio"
        )
        
        aspect_ratio = st.slider(
            "Aspect Ratio", 
            min_value=0.5, 
            max_value=4.0, 
            value=2.0, 
            step=0.5
        )
    
    with col2:
        st.markdown("**Numerical Parameters**")
        nx = st.slider("Grid Points X", min_value=20, max_value=128, value=64, step=8)
        ny = st.slider("Grid Points Y", min_value=20, max_value=128, value=64, step=8)
        max_iterations = st.slider("Max Iterations", min_value=100, max_value=10000, value=2000, step=100)
        tolerance = st.number_input("Tolerance", min_value=1e-10, max_value=1e-3, value=1e-6, format="%.0e")
    
    with col3:
        st.markdown("**ML Parameters**")
        training_epochs = st.slider("Training Epochs", min_value=1000, max_value=20000, value=5000, step=1000)
        learning_rate = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-2, value=1e-3, format="%.0e")
        
        # Display calculated parameters
        st.markdown("**Derived Parameters**")
        st.write(f"Critical Ra: 1708")
        st.write(f"Regime: {'Conduction' if Ra < 1708 else 'Convection'}")
        st.write(f"Grid Quality: {nx*ny} points")
    
    # Store parameters in session state
    st.session_state.update({
        'Ra': Ra, 'Pr': Pr, 'aspect_ratio': aspect_ratio,
        'nx': nx, 'ny': ny, 'max_iterations': max_iterations, 'tolerance': tolerance,
        'training_epochs': training_epochs, 'learning_rate': learning_rate
    })

with main_tabs[1]:
    st.header("🔬 Solution Methods")
    
    # Method selection
    col1, col2 = st.columns(2)
    
    with col1:
        method1 = st.selectbox(
            "Primary Method",
            options=['finite_difference', 'spectral', 'pinn_huggingface'],
            format_func=lambda x: SolverFactory.get_available_methods()[x]['name'],
            key="method1"
        )
    
    with col2:
        method2 = st.selectbox(
            "Comparison Method",
            options=['none', 'finite_difference', 'spectral', 'pinn_huggingface'],
            format_func=lambda x: 'None' if x == 'none' else SolverFactory.get_available_methods()[x]['name'],
            key="method2"
        )
    
    # Run buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        solve_method1 = st.button(
            f"🧮 Run {SolverFactory.get_available_methods()[method1]['name']}", 
            type="primary", 
            use_container_width=True
        )
    
    with col2:
        if method2 != 'none':
            solve_method2 = st.button(
                f"🧠 Run {SolverFactory.get_available_methods()[method2]['name']}", 
                type="primary", 
                use_container_width=True
            )
        else:
            solve_method2 = st.button("🚫 Run Comparison Method", disabled=True, use_container_width=True)
    
    with col3:
        compare_both = st.button("🔄 Compare Both Methods", type="secondary", use_container_width=True)
    
    # Initialize session state
    for key in ['results1', 'results2', 'time1', 'time2']:
        if key not in st.session_state:
            st.session_state[key] = None
    
    # Auto-compare if requested
    if compare_both:
        solve_method1 = True
        if method2 != 'none':
            solve_method2 = True
    
    # Run Method 1
    if solve_method1:
        st.subheader(f"🔬 {SolverFactory.get_available_methods()[method1]['name']}")
        
        if not SOLVERS_AVAILABLE:
            st.warning("⚠️ Advanced solvers not available on Streamlit Cloud. Showing demo results.")
            
            # Generate demo results
            x = np.linspace(0, 2, st.session_state.nx)
            y = np.linspace(0, 1, st.session_state.ny)
            X, Y = np.meshgrid(x, y)
            
            # Sample temperature field
            T = 1 - Y + 0.1 * np.sin(2*np.pi*X/2) * np.sin(np.pi*Y)
            
            # Sample velocity field
            psi = 0.1 * np.sin(np.pi * Y) * np.cos(2 * np.pi * X / 2)
            u = np.gradient(psi, axis=0)
            v = -np.gradient(psi, axis=1)
            
            results = {
                'x': x, 'y': y,
                'temperature': T,
                'stream_function': psi,
                'u_velocity': u,
                'v_velocity': v,
                'iterations': 100,
                'final_error': 1e-6
            }
            
            st.session_state.results1 = results
            st.session_state.time1 = 2.0  # Demo time
            
            st.success(f"✅ {method1} demo completed in {st.session_state.time1:.2f} seconds")
            
            # Show method details
            with st.expander("📖 Method Details"):
                st.json(SolverFactory.get_available_methods()[method1])
            
        else:
            with st.spinner(f"Running {method1}..."):
                try:
                    start_time = time.time()
                    solver = SolverFactory.create_solver(
                        method=method1,
                        nx=st.session_state.nx, ny=st.session_state.ny, 
                        Ra=st.session_state.Ra, Pr=st.session_state.Pr, 
                        aspect_ratio=st.session_state.aspect_ratio,
                        max_iter=st.session_state.max_iterations, 
                        tol=st.session_state.tolerance,
                        epochs=st.session_state.training_epochs, 
                        lr=st.session_state.learning_rate
                    )
                    results = solver.solve()
                    
                    st.session_state.results1 = results
                    st.session_state.time1 = time.time() - start_time
                    
                    st.success(f"✅ {method1} completed in {st.session_state.time1:.2f} seconds")
                    
                    # Show method details
                    with st.expander("📖 Method Details"):
                        st.json(SolverFactory.get_available_methods()[method1])
                    
                    # Validation
                    validation = validate_solution(results)
                    with st.expander("✅ Solution Validation"):
                        st.json(validation)
                    
                except Exception as e:
                    st.error(f"❌ {method1} failed: {str(e)}")
                    st.exception(e)
    
    # Run Method 2
    if method2 != 'none' and solve_method2:
        st.subheader(f"🧠 {SolverFactory.get_available_methods()[method2]['name']}")
        
        if not SOLVERS_AVAILABLE:
            st.warning("⚠️ Advanced solvers not available on Streamlit Cloud. Showing demo results.")
            
            # Generate demo results
            x = np.linspace(0, 2, st.session_state.nx)
            y = np.linspace(0, 1, st.session_state.ny)
            X, Y = np.meshgrid(x, y)
            
            # Sample temperature field (slightly different for demo)
            T = 1 - Y + 0.08 * np.sin(3*np.pi*X/2) * np.sin(np.pi*Y)
            
            # Sample velocity field
            psi = 0.12 * np.sin(np.pi * Y) * np.cos(3 * np.pi * X / 2)
            u = np.gradient(psi, axis=0)
            v = -np.gradient(psi, axis=1)
            
            results = {
                'x': x, 'y': y,
                'temperature': T,
                'stream_function': psi,
                'u_velocity': u,
                'v_velocity': v,
                'iterations': 120,
                'final_error': 8e-7
            }
            
            st.session_state.results2 = results
            st.session_state.time2 = 2.5  # Demo time
            
            st.success(f"✅ {method2} demo completed in {st.session_state.time2:.2f} seconds")
            
        else:
            with st.spinner(f"Running {method2}..."):
                try:
                    start_time = time.time()
                    solver = SolverFactory.create_solver(
                        method=method2,
                        nx=st.session_state.nx, ny=st.session_state.ny, 
                        Ra=st.session_state.Ra, Pr=st.session_state.Pr, 
                        aspect_ratio=st.session_state.aspect_ratio,
                        max_iter=st.session_state.max_iterations, 
                        tol=st.session_state.tolerance,
                        epochs=st.session_state.training_epochs, 
                        lr=st.session_state.learning_rate
                    )
                    results = solver.solve()
                    
                    st.session_state.results2 = results
                    st.session_state.time2 = time.time() - start_time
                    
                    st.success(f"✅ {method2} completed in {st.session_state.time2:.2f} seconds")
                    
                except Exception as e:
                    st.error(f"❌ {method2} failed: {str(e)}")
                    st.exception(e)

with main_tabs[2]:
    st.header("📊 Results Analysis & Comparison")
    
    if st.session_state.results1 is not None and st.session_state.results2 is not None:
        # Comprehensive comparison
        st.subheader("🔬 Comprehensive Solution Comparison")
        comparison_fig = create_solution_comparison(
            st.session_state.results1, st.session_state.results2, 
            method1, method2
        )
        if comparison_fig:
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Performance metrics
        st.markdown("---")
        st.subheader("📈 Performance Comparison")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"{method1} Time", f"{st.session_state.time1:.2f}s")
        with col2:
            st.metric(f"{method2} Time", f"{st.session_state.time2:.2f}s")
        with col3:
            speedup = st.session_state.time2 / st.session_state.time1
            st.metric("Speedup", f"{speedup:.2f}x")
        with col4:
            st.metric("Grid Points", f"{st.session_state.nx}×{st.session_state.ny}")
        
        # Error analysis
        st.markdown("---")
        st.subheader("📊 Error Analysis")
        
        # Calculate errors
        error_metrics = {}
        for field_name, field_key in [
            ('Temperature', 'temperature'),
            ('U Velocity', 'u_velocity'),
            ('V Velocity', 'v_velocity'),
            ('Stream Function', 'stream_function')
        ]:
            if field_key in st.session_state.results1 and field_key in st.session_state.results2:
                field1 = st.session_state.results1[field_key]
                field2 = st.session_state.results2[field_key]
                
                if not (np.any(np.isnan(field1)) or np.any(np.isnan(field2))):
                    l2_error = np.sqrt(np.mean((field1 - field2)**2))
                    rel_error = l2_error / (np.sqrt(np.mean(field1**2)) + 1e-10)
                    max_error = np.max(np.abs(field1 - field2))
                    
                    error_metrics[field_name] = {
                        'L2 Error': l2_error,
                        'Relative Error': rel_error,
                        'Max Error': max_error
                    }
        
        if error_metrics:
            # Create error comparison chart
            fig = go.Figure()
            
            fields = list(error_metrics.keys())
            l2_errors = [error_metrics[f]['L2 Error'] for f in fields]
            rel_errors = [error_metrics[f]['Relative Error'] for f in fields]
            max_errors = [error_metrics[f]['Max Error'] for f in fields]
            
            fig.add_trace(go.Bar(name='L2 Error', x=fields, y=l2_errors))
            fig.add_trace(go.Bar(name='Relative Error', x=fields, y=rel_errors))
            fig.add_trace(go.Bar(name='Max Error', x=fields, y=max_errors))
            
            fig.update_layout(
                title='Error Metrics Comparison',
                xaxis_title='Field',
                yaxis_title='Error Magnitude',
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed error table
            st.json(error_metrics)
        
        # Physical analysis
        st.markdown("---")
        st.subheader("🌊 Physical Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Nusselt number calculation
            if 'temperature' in st.session_state.results1:
                Nu1 = compute_nusselt_number(
                    st.session_state.results1['temperature'],
                    st.session_state.results1['y']
                )
                st.metric(f"{method1} Nusselt Number", f"{Nu1:.3f}")
            
            if 'temperature' in st.session_state.results2:
                Nu2 = compute_nusselt_number(
                    st.session_state.results2['temperature'],
                    st.session_state.results2['y']
                )
                st.metric(f"{method2} Nusselt Number", f"{Nu2:.3f}")
        
        with col2:
            # Flow statistics
            if 'u_velocity' in st.session_state.results1:
                u1 = st.session_state.results1['u_velocity']
                v1 = st.session_state.results1['v_velocity']
                speed1 = np.sqrt(u1**2 + v1**2)
                st.metric(f"{method1} Max Velocity", f"{np.max(speed1):.3f}")
            
            if 'u_velocity' in st.session_state.results2:
                u2 = st.session_state.results2['u_velocity']
                v2 = st.session_state.results2['v_velocity']
                speed2 = np.sqrt(u2**2 + v2**2)
                st.metric(f"{method2} Max Velocity", f"{np.max(speed2):.3f}")
    
    else:
        st.info("🔄 Please run at least one method to see results analysis")

with main_tabs[3]:
    st.header("🔄 Comprehensive Method Comparison")
    
    st.markdown("""
    This section provides detailed comparison between different numerical methods for solving 
    Rayleigh-Bénard convection. Compare accuracy, performance, and physical consistency.
    """)
    
    # Check if we have results to compare
    if st.session_state.results1 is not None and st.session_state.results2 is not None:
        st.success("✅ Both methods completed! Ready for comprehensive comparison.")
        
        # Method selection for comparison
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Method 1:** {method1.replace('_', ' ').title()}")
            if st.session_state.time1:
                st.write(f"⏱️ Computation time: {st.session_state.time1:.2f}s")
            if st.session_state.results1:
                st.write(f"🔄 Iterations: {st.session_state.results1.get('iterations', 'N/A')}")
                st.write(f"📊 Final error: {st.session_state.results1.get('final_error', 0):.2e}")
        
        with col2:
            st.markdown(f"**Method 2:** {method2.replace('_', ' ').title()}")
            if st.session_state.time2:
                st.write(f"⏱️ Computation time: {st.session_state.time2:.2f}s")
            if st.session_state.results2:
                st.write(f"🔄 Iterations: {st.session_state.results2.get('iterations', 'N/A')}")
                st.write(f"📊 Final error: {st.session_state.results2.get('final_error', 0):.2e}")
        
        # Comprehensive comparison plots
        st.markdown("---")
        st.subheader("🔬 Side-by-Side Solution Comparison")
        
        comparison_fig = create_solution_comparison(
            st.session_state.results1, st.session_state.results2, 
            method1, method2
        )
        if comparison_fig:
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Detailed error analysis
        st.markdown("---")
        st.subheader("📊 Quantitative Error Analysis")
        
        # Calculate and display errors
        error_metrics = {}
        for field_name, field_key in [
            ('Temperature', 'temperature'),
            ('U Velocity', 'u_velocity'),
            ('V Velocity', 'v_velocity'),
            ('Stream Function', 'stream_function')
        ]:
            if field_key in st.session_state.results1 and field_key in st.session_state.results2:
                field1 = st.session_state.results1[field_key]
                field2 = st.session_state.results2[field_key]
                
                if not (np.any(np.isnan(field1)) or np.any(np.isnan(field2))):
                    l2_error = np.sqrt(np.mean((field1 - field2)**2))
                    rel_error = l2_error / (np.sqrt(np.mean(field1**2)) + 1e-10)
                    max_error = np.max(np.abs(field1 - field2))
                    
                    error_metrics[field_name] = {
                        'L2 Error': l2_error,
                        'Relative Error': rel_error,
                        'Max Error': max_error
                    }
        
        if error_metrics:
            # Create error comparison chart
            fig = go.Figure()
            
            fields = list(error_metrics.keys())
            l2_errors = [error_metrics[f]['L2 Error'] for f in fields]
            rel_errors = [error_metrics[f]['Relative Error'] for f in fields]
            max_errors = [error_metrics[f]['Max Error'] for f in fields]
            
            fig.add_trace(go.Bar(name='L2 Error', x=fields, y=l2_errors))
            fig.add_trace(go.Bar(name='Relative Error', x=fields, y=rel_errors))
            fig.add_trace(go.Bar(name='Max Error', x=fields, y=max_errors))
            
            fig.update_layout(
                title='Error Metrics Comparison Between Methods',
                xaxis_title='Field',
                yaxis_title='Error Magnitude',
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed error table
            st.subheader("📋 Detailed Error Metrics")
            for field_name, metrics in error_metrics.items():
                with st.expander(f"{field_name} Error Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("L2 Error", f"{metrics['L2 Error']:.4e}")
                    with col2:
                        st.metric("Relative Error", f"{metrics['Relative Error']:.4e}")
                    with col3:
                        st.metric("Max Error", f"{metrics['Max Error']:.4e}")
        
        # Physical comparison
        st.markdown("---")
        st.subheader("🌊 Physical Quantities Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Heat Transfer Analysis**")
            if 'temperature' in st.session_state.results1:
                Nu1 = compute_nusselt_number(
                    st.session_state.results1['temperature'],
                    st.session_state.results1['y']
                )
                st.metric(f"{method1} Nusselt Number", f"{Nu1:.3f}")
                
                if Nu1 > 1.0:
                    st.success(f"✅ Convective heat transfer enhancement: {((Nu1-1)*100):.1f}%")
                else:
                    st.info("ℹ️ Conduction-dominated heat transfer")
            
            if 'temperature' in st.session_state.results2:
                Nu2 = compute_nusselt_number(
                    st.session_state.results2['temperature'],
                    st.session_state.results2['y']
                )
                st.metric(f"{method2} Nusselt Number", f"{Nu2:.3f}")
                
                if Nu2 > 1.0:
                    st.success(f"✅ Convective heat transfer enhancement: {((Nu2-1)*100):.1f}%")
        
        with col2:
            st.markdown("**Flow Characteristics**")
            if 'u_velocity' in st.session_state.results1:
                u1 = st.session_state.results1['u_velocity']
                v1 = st.session_state.results1['v_velocity']
                speed1 = np.sqrt(u1**2 + v1**2)
                st.metric(f"{method1} Max Velocity", f"{np.max(speed1):.3f}")
                st.metric(f"{method1} Mean Velocity", f"{np.mean(speed1):.3f}")
            
            if 'u_velocity' in st.session_state.results2:
                u2 = st.session_state.results2['u_velocity']
                v2 = st.session_state.results2['v_velocity']
                speed2 = np.sqrt(u2**2 + v2**2)
                st.metric(f"{method2} Max Velocity", f"{np.max(speed2):.3f}")
                st.metric(f"{method2} Mean Velocity", f"{np.mean(speed2):.3f}")
        
        # Performance comparison
        st.markdown("---")
        st.subheader("⚡ Performance Comparison")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"{method1} Time", f"{st.session_state.time1:.2f}s")
        with col2:
            st.metric(f"{method2} Time", f"{st.session_state.time2:.2f}s")
        with col3:
            if st.session_state.time1 > 0 and st.session_state.time2 > 0:
                speedup = st.session_state.time2 / st.session_state.time1
                faster_method = method1 if speedup > 1 else method2
                st.metric("Speedup", f"{abs(speedup):.2f}x", delta=f"{faster_method} faster")
        with col4:
            st.metric("Grid Resolution", f"{st.session_state.nx}×{st.session_state.ny}")
        
        # Method recommendations
        st.markdown("---")
        st.subheader("💡 Method Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{method1.replace('_', ' ').title()}**")
            if method1 == 'finite_difference':
                st.info("✅ **Best for:**\n- General purpose problems\n- Complex geometries\n- Robustness is critical")
            elif method1 == 'spectral':
                st.info("✅ **Best for:**\n- High accuracy requirements\n- Smooth solutions\n- Research applications")
            elif method1 == 'pinn_huggingface':
                st.info("✅ **Best for:**\n- Transfer learning\n- Parameter studies\n- Mesh-free solutions")
        
        with col2:
            st.markdown(f"**{method2.replace('_', ' ').title()}**")
            if method2 == 'finite_difference':
                st.info("✅ **Best for:**\n- General purpose problems\n- Complex geometries\n- Robustness is critical")
            elif method2 == 'spectral':
                st.info("✅ **Best for:**\n- High accuracy requirements\n- Smooth solutions\n- Research applications")
            elif method2 == 'pinn_huggingface':
                st.info("✅ **Best for:**\n- Transfer learning\n- Parameter studies\n- Mesh-free solutions")
        
    elif st.session_state.results1 is not None or st.session_state.results2 is not None:
        available_method = method1 if st.session_state.results1 is not None else method2
        st.info(f"🔄 Only {available_method.replace('_', ' ').title()} has been run. Run another method to enable comparison.")
        
        # Show available results
        if st.session_state.results1 is not None:
            st.subheader(f"📊 {method1.replace('_', ' ').title()} Results")
            st.write("Run a second method to see detailed comparison.")
        
        if st.session_state.results2 is not None:
            st.subheader(f"📊 {method2.replace('_', ' ').title()} Results")
            st.write("Run a second method to see detailed comparison.")
    
    else:
        st.warning("🔄 No methods have been run yet. Go to **Solution Methods** tab to run solvers and enable comparison.")
        
        # Show method comparison guide
        st.markdown("---")
        st.subheader("📖 How to Compare Methods")
        
        st.markdown("""
        **Step 1:** Go to **🔬 Solution Methods** tab
        
        **Step 2:** Select two different methods to compare:
        - **Finite Difference** - Traditional CFD approach
        - **Spectral Method** - High accuracy Fourier method  
        - **PINN** - Machine learning approach
        
        **Step 3:** Run both methods (individually or use "Compare Both")
        
        **Step 4:** Return to this **🔄 Method Comparison** tab for detailed analysis
        """)
        
        # Show available methods
        methods = SolverFactory.get_available_methods()
        st.markdown("**Available Methods:**")
        for method_key, method_info in methods.items():
            with st.expander(f"🔬 {method_info['name']}"):
                st.write(f"**Description:** {method_info['description']}")
                st.write("**Advantages:**")
                for pro in method_info['pros']:
                    st.write(f"✅ {pro}")
                st.write("**Limitations:**")
                for con in method_info['cons']:
                    st.write(f"❌ {con}")

with main_tabs[4]:
    st.header("📚 Theory & Background")
    
    # Comprehensive theory section
    st.markdown("""
    ## 🌊 Rayleigh-Bénard Convection Theory
    
    ### Historical Background
    Rayleigh-Bénard convection is named after Lord Rayleigh (1916) and Henri Bénard (1900), 
    who independently studied the instability of a fluid layer heated from below.
    
    ### Physical Mechanism
    When a fluid layer is heated from below and cooled from above, a temperature gradient 
    is established. If the temperature difference is small, heat is transferred by conduction 
    alone. However, when the temperature difference exceeds a critical value, the fluid becomes 
    unstable and convection cells form.
    
    ### Linear Stability Analysis
    The critical Rayleigh number for the onset of convection is:
    $$Ra_c = \\frac{27\\pi^4}{4} \\approx 1708$$
    
    This occurs for a wavelength of: $$\\lambda_c = 2\\sqrt{2}H$$
    
    where H is the height of the fluid layer.
    
    ### Nonlinear Dynamics
    Above the critical Rayleigh number, the system exhibits rich dynamics:
    - **Steady convection rolls** (Ra < 10⁴)
    - **Oscillatory convection** (10⁴ < Ra < 10⁶)
    - **Chaotic convection** (10⁶ < Ra < 10⁸)
    - **Turbulent convection** (Ra > 10⁸)
    
    ### Applications
    - **Geophysics**: Mantle convection, atmospheric circulation
    - **Engineering**: Heat exchangers, cooling systems
    - **Astrophysics**: Stellar convection zones
    - **Climate science**: Oceanic and atmospheric dynamics
    
    ### Numerical Methods Comparison
    
    **Finite Difference Method:**
    - Direct discretization of governing equations
    - Second-order accuracy in space
    - Easy to implement for complex geometries
    - May suffer from numerical diffusion
    
    **Spectral Method:**
    - Expansion in orthogonal functions (Fourier series)
    - Exponential convergence for smooth solutions
    - Minimal numerical diffusion
    - Best for periodic boundary conditions
    
    **Physics-Informed Neural Networks:**
    - Machine learning approach with physics constraints
    - Mesh-free method
    - Can handle complex boundary conditions
    - Training can be computationally expensive
    """)

# Footer
st.markdown("---")
st.markdown("""
**🔬 Systematic Computational Fluid Dynamics Laboratory**  
*Complete Rayleigh-Bénard convection analysis from problem setup to solution comparison*
""")
