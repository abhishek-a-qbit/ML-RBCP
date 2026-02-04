"""
Enhanced Convection Solver App with Robust Backend
Multiple numerical methods, ML integration, and comprehensive explanations
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os

# Add the current directory to Python path to import our solvers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from solvers import SolverFactory, compute_nusselt_number, validate_solution
except ImportError as e:
    st.error(f"Error importing solvers: {e}")
    st.stop()

# ============================================================================
# ENHANCED VISUALIZATION FUNCTIONS
# ============================================================================

def create_field_plot(results, field_name, method):
    """Create a single field plot with both Plotly and Matplotlib options"""
    if field_name not in results or results[field_name] is None:
        # Create empty plot with error message
        fig = go.Figure()
        fig.add_annotation(text="Data not available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    field_data = results[field_name]
    
    # Check for NaN or invalid data
    if np.any(np.isnan(field_data)) or np.any(np.isinf(field_data)):
        # Create plot showing data issues
        fig = go.Figure()
        fig.add_annotation(text="Invalid data (NaN/Inf detected)", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Determine color scale and range
    if field_name == 'Temperature':
        colorscale = 'RdBu_r'
        zmin, zmax = 0, 1
    elif field_name == 'Stream Function':
        colorscale = 'Viridis'
        zmin, zmax = np.min(field_data), np.max(field_data)
    elif field_name == 'U Velocity':
        colorscale = 'RdBu_r'
        vmax = np.max(np.abs(field_data))
        zmin, zmax = -vmax, vmax
    else:  # V Velocity
        colorscale = 'RdBu_r'
        vmax = np.max(np.abs(field_data))
        zmin, zmax = -vmax, vmax
    
    fig = go.Figure(data=go.Heatmap(
        z=field_data,
        x=results['x'],
        y=results['y'],
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title=field_name)
    ))
    
    fig.update_layout(
        title=f"{field_name} - {results.get('method', method).replace('_', ' ').title()}",
        xaxis_title="x",
        yaxis_title="y",
        height=500,
        width=800
    )
    
    return fig

def create_matplotlib_visualization(results, field_name):
    """Create matplotlib figure for additional analysis"""
    import matplotlib.pyplot as plt
    
    if field_name not in results or results[field_name] is None:
        return None
    
    field_data = results[field_name]
    
    if np.any(np.isnan(field_data)) or np.any(np.isinf(field_data)):
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap
    im1 = ax1.contourf(results['X'], results['Y'], field_data, levels=20, cmap='RdBu_r')
    ax1.set_title(f"{field_name} - Contour Plot")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)
    
    # Line plots at different y positions
    mid_y = len(results['y']) // 2
    mid_y_val = results['y'][mid_y]
    ax2.plot(results['x'], field_data[mid_y, :], 'b-', linewidth=2, label=f'y = {mid_y_val:.2f}')
    
    if len(results['y']) > 4:
        quarter_y = len(results['y']) // 4
        quarter_y_val = results['y'][quarter_y]
        ax2.plot(results['x'], field_data[quarter_y, :], 'r--', linewidth=2, label=f'y = {quarter_y_val:.2f}')
        three_quarter_y = 3 * len(results['y']) // 4
        three_quarter_y_val = results['y'][three_quarter_y]
        ax2.plot(results['x'], field_data[three_quarter_y, :], 'g--', linewidth=2, label=f'y = {three_quarter_y_val:.2f}')
    
    ax2.set_title(f"{field_name} - Horizontal Profiles")
    ax2.set_xlabel("x")
    ax2.set_ylabel(field_name)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_velocity_field_plot(results):
    """Create velocity field visualization with quiver plot"""
    if 'u_velocity' not in results or 'v_velocity' not in results:
        return None
    
    u = results['u_velocity']
    v = results['v_velocity']
    
    if np.any(np.isnan(u)) or np.any(np.isnan(v)):
        return None
    
    # Create subplot for velocity magnitude and vectors
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Velocity Magnitude', 'Velocity Vectors'),
        horizontal_spacing=0.15
    )
    
    # Velocity magnitude
    speed = np.sqrt(u**2 + v**2)
    fig.add_trace(
        go.Heatmap(z=speed, x=results['x'], y=results['y'], 
                   colorscale='Viridis', colorbar=dict(title="Speed", x=1.15)),
        row=1, col=1
    )
    
    # Velocity vectors (quiver plot)
    skip = max(1, len(results['x']) // 20)  # Skip points for clarity
    x_sub = results['x'][::skip]
    y_sub = results['y'][::skip]
    u_sub = u[::skip, ::skip]
    v_sub = v[::skip, ::skip]
    
    fig.add_trace(
        go.Scatter(
            x=x_sub.repeat(len(y_sub)),
            y=np.tile(y_sub, len(x_sub)),
            mode='markers',
            marker=dict(
                size=3,
                color=np.sqrt(u_sub.flatten()**2 + v_sub.flatten()**2),
                colorscale='Viridis',
                showscale=False
            )
        ),
        row=1, col=2
    )
    
    # Add velocity vectors as lines
    for i in range(len(x_sub)):
        for j in range(len(y_sub)):
            fig.add_shape(
                type="line",
                x0=x_sub[i], y0=y_sub[j],
                x1=x_sub[i] + u_sub[j, i]*0.1, y1=y_sub[j] + v_sub[j, i]*0.1,
                line=dict(color="black", width=1),
                row=1, col=2
            )
    
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=2)
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=2)
    fig.update_layout(height=500, title_text="Velocity Field Analysis")
    
    return fig

def create_comparison_plot(results1, results2, field_name, method1, method2):
    """Create side-by-side comparison plot"""
    field_map = {
        'Temperature': 'temperature',
        'Stream Function': 'stream_function', 
        'U Velocity': 'u_velocity',
        'V Velocity': 'v_velocity'
    }
    field_key = field_map[field_name]
    
    # Check if data is available
    if field_key not in results1 or field_key not in results2:
        fig = go.Figure()
        fig.add_annotation(text="Data not available for comparison", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    field1 = results1[field_key]
    field2 = results2[field_key]
    
    # Check for invalid data
    if np.any(np.isnan(field1)) or np.any(np.isnan(field2)):
        fig = go.Figure()
        fig.add_annotation(text="Invalid data detected", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f"{method1.replace('_', ' ').title()}", 
            f"{method2.replace('_', ' ').title()}", 
            "Absolute Difference"
        ),
        horizontal_spacing=0.1
    )
    
    # Determine common color scale
    vmin = min(np.min(field1), np.min(field2))
    vmax = max(np.max(field1), np.max(field2))
    
    # Plot first method
    fig.add_trace(
        go.Heatmap(z=field1, x=results1['x'], y=results1['y'], 
                   colorscale='RdBu_r', zmin=vmin, zmax=vmax, showscale=False),
        row=1, col=1
    )
    
    # Plot second method
    fig.add_trace(
        go.Heatmap(z=field2, x=results2['x'], y=results2['y'], 
                   colorscale='RdBu_r', zmin=vmin, zmax=vmax, showscale=False),
        row=1, col=2
    )
    
    # Plot difference
    diff = np.abs(field1 - field2)
    fig.add_trace(
        go.Heatmap(z=diff, x=results1['x'], y=results1['y'], 
                   colorscale='Hot', showscale=True,
                   colorbar=dict(title="|Difference|", x=1.15)),
        row=1, col=3
    )
    
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=2)
    fig.update_xaxes(title_text="x", row=1, col=3)
    fig.update_yaxes(title_text="y", row=1, col=1)
    
    fig.update_layout(height=500, title_text=f"{field_name} Comparison")
    
    return fig

def compute_error_metrics(results1, results2):
    """Compute error metrics between two solutions"""
    metrics = {}
    
    for field_name, field_key in [
        ('Temperature', 'temperature'),
        ('U Velocity', 'u_velocity'),
        ('V Velocity', 'v_velocity'),
        ('Stream Function', 'stream_function')
    ]:
        if field_key in results1 and field_key in results2:
            field1 = results1[field_key]
            field2 = results2[field_key]
            
            # Skip if either has NaN
            if np.any(np.isnan(field1)) or np.any(np.isnan(field2)):
                metrics[field_name] = {
                    'L2 Error': np.nan,
                    'Relative L2 Error': np.nan,
                    'Max Error': np.nan,
                    'Mean Absolute Error': np.nan,
                    'Note': 'Invalid data detected'
                }
                continue
            
            # L2 error
            l2_error = np.sqrt(np.mean((field1 - field2)**2))
            
            # Relative L2 error
            rel_l2_error = l2_error / (np.sqrt(np.mean(field1**2)) + 1e-10)
            
            # Max absolute error
            max_error = np.max(np.abs(field1 - field2))
            
            # Mean absolute error
            mae = np.mean(np.abs(field1 - field2))
            
            metrics[field_name] = {
                'L2 Error': l2_error,
                'Relative L2 Error': rel_l2_error,
                'Max Error': max_error,
                'Mean Absolute Error': mae
            }
    
    return metrics

# Set page config
st.set_page_config(
    page_title="Advanced Rayleigh-Bénard Convection Solver", 
    layout="wide", 
    page_icon="🌊"
)

# Title and description
st.title("🌊 Advanced Rayleigh-Bénard Convection Solver")
st.markdown("""
Comprehensive solver using **multiple numerical methods** and **machine learning approaches**. 
Compare traditional CFD with modern ML techniques, validate solutions, and explore convection physics!
""")

# Sidebar for parameters
st.sidebar.header("⚙️ Problem Parameters")

# Physical parameters
Ra = st.sidebar.number_input(
    "Rayleigh Number (Ra)", 
    min_value=1e3, 
    max_value=1e6, 
    value=1e4, 
    format="%.0e",
    help="Ratio of buoyancy to viscous forces. Higher Ra = stronger convection"
)

Pr = st.sidebar.number_input(
    "Prandtl Number (Pr)", 
    min_value=0.01, 
    max_value=100.0, 
    value=0.71,
    help="Ratio of momentum diffusivity to thermal diffusivity"
)

aspect_ratio = st.sidebar.slider(
    "Aspect Ratio", 
    min_value=0.5, 
    max_value=4.0, 
    value=2.0, 
    step=0.5,
    help="Width to height ratio of the domain"
)

st.sidebar.markdown("---")
st.sidebar.header("🔧 Numerical Parameters")

# Grid resolution
nx = st.sidebar.slider("Grid Points (X)", min_value=20, max_value=128, value=64, step=8)
ny = st.sidebar.slider("Grid Points (Y)", min_value=20, max_value=128, value=64, step=8)
max_iterations = st.sidebar.slider("Max Iterations", min_value=100, max_value=10000, value=2000, step=100)
tolerance = st.sidebar.number_input(
    "Convergence Tolerance", 
    min_value=1e-10, 
    max_value=1e-3, 
    value=1e-6, 
    format="%.0e"
)

st.sidebar.markdown("---")
st.sidebar.header("🧠 ML Parameters")

# ML parameters
training_epochs = st.sidebar.slider("Training Epochs", min_value=1000, max_value=20000, value=5000, step=1000)
learning_rate = st.sidebar.number_input(
    "Learning Rate", 
    min_value=1e-5, 
    max_value=1e-2, 
    value=1e-3, 
    format="%.0e"
)

# Add helpful tips
st.sidebar.markdown("---")
st.sidebar.info("💡 **Recommendations:**\n- Start with Ra = 1e4\n- Use 64×64 grid for accuracy\n- Compare multiple methods")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Solve & Compare", "📊 Analysis", "📚 Theory", "⚙️ Methods"])

with tab1:
    st.header("Solver Comparison & Results")
    
    # Method selection
    col1, col2, col3 = st.columns(3)
    
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
    
    with col3:
        auto_run = st.checkbox("Auto-run both methods", value=True)
    
    # Run buttons
    col1, col2 = st.columns(2)
    
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
    
    # Initialize session state
    for key in ['results1', 'results2', 'time1', 'time2']:
        if key not in st.session_state:
            st.session_state[key] = None
    
    # Auto-run if enabled
    if auto_run and (solve_method1 or (method2 != 'none' and solve_method2)):
        if solve_method1:
            solve_method2 = True
    
    # Run Method 1
    if solve_method1:
        st.subheader(f"🔬 {SolverFactory.get_available_methods()[method1]['name']}")
        start_time = time.time()
        
        try:
            with st.spinner(f"Running {method1}..."):
                solver = SolverFactory.create_solver(
                    method=method1,
                    nx=nx, ny=ny, Ra=Ra, Pr=Pr, aspect_ratio=aspect_ratio,
                    max_iter=max_iterations, tol=tolerance,
                    epochs=training_epochs, lr=learning_rate
                )
                results = solver.solve()
            
            st.session_state.results1 = results
            st.session_state.time1 = time.time() - start_time
            st.success(f"✅ {method1} completed in {st.session_state.time1:.2f} seconds")
            
            # Show method info
            with st.expander("📖 Method Information"):
                st.json(SolverFactory.get_available_methods()[method1])
            
            # Show equations
            with st.expander("🧮 Governing Equations"):
                if 'equations' in results:
                    for key, value in results['equations'].items():
                        st.write(f"**{key}:** {value}")
            
            # Show validation
            with st.expander("✅ Solution Validation"):
                validation = validate_solution(results)
                st.json(validation)
            
        except Exception as e:
            st.error(f"❌ {method1} failed: {str(e)}")
            st.exception(e)
    
    # Run Method 2
    if method2 != 'none' and solve_method2:
        st.subheader(f"🧠 {SolverFactory.get_available_methods()[method2]['name']}")
        start_time = time.time()
        
        try:
            with st.spinner(f"Running {method2}..."):
                solver = SolverFactory.create_solver(
                    method=method2,
                    nx=nx, ny=ny, Ra=Ra, Pr=Pr, aspect_ratio=aspect_ratio,
                    max_iter=max_iterations, tol=tolerance,
                    epochs=training_epochs, lr=learning_rate
                )
                results = solver.solve()
            
            st.session_state.results2 = results
            st.session_state.time2 = time.time() - start_time
            st.success(f"✅ {method2} completed in {st.session_state.time2:.2f} seconds")
            
            # Show method info
            with st.expander("📖 Method Information"):
                st.json(SolverFactory.get_available_methods()[method2])
            
        except Exception as e:
            st.error(f"❌ {method2} failed: {str(e)}")
            st.exception(e)
    
    # Display individual results
    if st.session_state.results1 is not None:
        st.markdown("---")
        st.subheader(f"📈 {SolverFactory.get_available_methods()[method1]['name']} Results")
        
        # Field selector and visualization options
        col1, col2 = st.columns([2, 1])
        with col1:
            field_to_show = st.selectbox(
                "Select field to view:",
                ['Temperature', 'Stream Function', 'U Velocity', 'V Velocity'],
                key="field1"
            )
        
        with col2:
            viz_type = st.selectbox(
                "Visualization:",
                ['Plotly Heatmap', 'Matplotlib Analysis', 'Velocity Field'],
                key="viz_type1"
            )
        
        # Create appropriate visualization
        if viz_type == 'Plotly Heatmap':
            fig = create_field_plot(st.session_state.results1, field_to_show, method1)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == 'Matplotlib Analysis':
            matplotlib_fig = create_matplotlib_visualization(st.session_state.results1, field_to_show)
            if matplotlib_fig is not None:
                st.pyplot(matplotlib_fig)
            else:
                st.warning("Matplotlib visualization not available for this field")
        
        elif viz_type == 'Velocity Field' and field_to_show in ['U Velocity', 'V Velocity']:
            vel_fig = create_velocity_field_plot(st.session_state.results1)
            if vel_fig is not None:
                st.plotly_chart(vel_fig, use_container_width=True)
            else:
                st.warning("Velocity field visualization not available")
        else:
            # Default to heatmap
            fig = create_field_plot(st.session_state.results1, field_to_show, method1)
            st.plotly_chart(fig, use_container_width=True)
        
        # Show metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Iterations", st.session_state.results1.get('iterations', 'N/A'))
        with col2:
            st.metric("Final Error", f"{st.session_state.results1.get('final_error', 0):.2e}")
        with col3:
            st.metric("Compute Time", f"{st.session_state.time1:.2f}s")
    
    if st.session_state.results2 is not None:
        st.markdown("---")
        st.subheader(f"🧠 {SolverFactory.get_available_methods()[method2]['name']} Results")
        
        # Field selector
        field_to_show = st.selectbox(
            "Select field to view:",
            ['Temperature', 'Stream Function', 'U Velocity', 'V Velocity'],
            key="field2"
        )
        
        # Create plot
        fig = create_field_plot(st.session_state.results2, field_to_show, method2)
        st.plotly_chart(fig, use_container_width=True)
    
    # Comparison view
    if st.session_state.results1 is not None and st.session_state.results2 is not None:
        st.markdown("---")
        st.header("📊 Direct Comparison")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{method1} Time", f"{st.session_state.time1:.2f}s")
        with col2:
            st.metric(f"{method2} Time", f"{st.session_state.time2:.2f}s")
        with col3:
            speedup = st.session_state.time2 / st.session_state.time1
            st.metric("Speedup", f"{speedup:.2f}x", 
                     delta=f"{method1} faster" if speedup > 1 else f"{method2} faster")
        
        # Side-by-side comparison
        field_to_compare = st.selectbox(
            "Select field to compare:",
            ['Temperature', 'Stream Function', 'U Velocity', 'V Velocity'],
            key="compare_field"
        )
        
        fig = create_comparison_plot(
            st.session_state.results1, st.session_state.results2, 
            field_to_compare, method1, method2
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Error analysis
        with st.expander("📊 Error Analysis"):
            error_metrics = compute_error_metrics(st.session_state.results1, st.session_state.results2)
            st.json(error_metrics)

with tab2:
    st.header("📊 Analysis & Diagnostics")
    
    if st.session_state.results1 is not None:
        st.subheader("🔬 Solution Analysis")
        
        # Nusselt number calculation
        if 'temperature' in st.session_state.results1:
            Nu = compute_nusselt_number(
                st.session_state.results1['temperature'],
                st.session_state.results1['y']
            )
            st.metric("Nusselt Number", f"{Nu:.3f}")
            st.info("Nusselt number > 1 indicates convective heat transfer enhancement")
        
        # Validation results
        validation = validate_solution(st.session_state.results1)
        st.subheader("✅ Physical Validation")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Temperature Bounds:**")
            temp_ok = validation['temperature_bounds_ok']
            st.write(f"✅ Valid: {temp_ok}")
            st.write(f"Range: {validation['T_range']}")
        
        with col2:
            st.write("**Incompressibility:**")
            div_ok = validation['divergence_ok']
            st.write(f"✅ Valid: {div_ok}")
            if validation['divergence'] is not None:
                st.write(f"Mean |∇·u|: {validation['divergence']:.2e}")
        
        # Flow statistics
        if 'u_velocity' in st.session_state.results1 and 'v_velocity' in st.session_state.results1:
            u = st.session_state.results1['u_velocity']
            v = st.session_state.results1['v_velocity']
            speed = np.sqrt(u**2 + v**2)
            
            st.subheader("🌊 Flow Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max Velocity", f"{np.max(speed):.3f}")
            with col2:
                st.metric("Mean Velocity", f"{np.mean(speed):.3f}")
            with col3:
                st.metric("RMS Velocity", f"{np.sqrt(np.mean(speed**2)):.3f}")

with tab3:
    st.header("📚 Theory & Background")
    
    st.markdown("""
    ## 🌊 Rayleigh-Bénard Convection
    
    Rayleigh-Bénard convection is a fundamental fluid dynamics phenomenon where a fluid layer 
    is heated from below and cooled from above, leading to the formation of convection cells.
    
    ### 🔬 Governing Equations
    
    The phenomenon is governed by the **Boussinesq approximation** of the Navier-Stokes equations:
    
    **Continuity Equation (Incompressibility):**
    ```
    ∇·u = 0
    ```
    
    **Momentum Equations:**
    ```
    ∂u/∂t + (u·∇)u = -∇p + Pr·∇²u
    ∂v/∂t + (u·∇)v = -∇p + Pr·∇²v + Ra·Pr·T·ĵ
    ```
    
    **Energy Equation:**
    ```
    ∂T/∂t + (u·∇)T = ∇²T
    ```
    
    ### 📊 Key Parameters
    
    - **Rayleigh Number (Ra)**: `Ra = g·α·ΔT·L³/(ν·κ)`
      - Ratio of buoyancy to viscous forces
      - Ra < 1708: Conduction (no flow)
      - Ra > 1708: Convection begins
    
    - **Prandtl Number (Pr)**: `Pr = ν/κ`
      - Ratio of momentum to thermal diffusivity
      - Air: Pr ≈ 0.71, Water: Pr ≈ 7.0
    
    - **Nusselt Number (Nu)**: `Nu = Q_actual/Q_conduction`
      - Ratio of actual to conductive heat transfer
      - Nu = 1: Pure conduction
      - Nu > 1: Convective enhancement
    
    ### 🌡️ Physical Regimes
    
    | Ra Range | Flow Regime | Characteristics |
    |----------|-------------|----------------|
    | < 1708 | Conduction | No fluid motion, linear temperature profile |
    | 1708-10⁴ | Onset of Convection | Steady convection rolls |
    | 10⁴-10⁶ | Laminar Convection | Well-defined convection patterns |
    | > 10⁶ | Turbulent Convection | Chaotic, time-dependent flow |
    
    ### 🔢 Numerical Methods
    
    **Finite Difference Method:**
    - Discretizes domain into grid points
    - Approximates derivatives with finite differences
    - Pros: Simple, robust, easy to implement
    - Cons: Numerical diffusion, requires fine grids
    
    **Spectral Method:**
    - Represents solution as Fourier series
    - Exact derivatives in spectral space
    - Pros: Exponential convergence, no numerical diffusion
    - Cons: Requires periodic boundaries, Gibbs phenomena
    
    **Physics-Informed Neural Networks:**
    - Neural networks learn solution directly
    - Physics enforced through loss function
    - Pros: Mesh-free, transfer learning, differentiable
    - Cons: Training intensive, convergence not guaranteed
    """)

with tab4:
    st.header("⚙️ Available Methods")
    
    methods = SolverFactory.get_available_methods()
    
    for method_key, method_info in methods.items():
        with st.expander(f"🔬 {method_info['name']}"):
            st.write(f"**Description:** {method_info['description']}")
            
            st.write("**Advantages:**")
            for pro in method_info['pros']:
                st.write(f"✅ {pro}")
            
            st.write("**Limitations:**")
            for con in method_info['cons']:
                st.write(f"❌ {con}")
            
            st.write("**Best for:**")
            if method_key == 'finite_difference':
                st.write("- General purpose problems")
                st.write("- Complex geometries")
                st.write("- Robustness is critical")
            elif method_key == 'spectral':
                st.write("- Smooth solutions")
                st.write("- High accuracy requirements")
                st.write("- Periodic boundary conditions")
            elif method_key == 'pinn_huggingface':
                st.write("- Transfer learning applications")
                st.write("- Parameter studies")
                st.write("- Mesh-free solutions")

# Footer
st.markdown("---")
st.markdown("""
**🔬 Advanced Computational Fluid Dynamics Laboratory**  
*Bridging traditional numerical methods with modern machine learning for fluid mechanics research*
""")
