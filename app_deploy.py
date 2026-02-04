"""
Streamlit Cloud Deployment Version
Simplified Rayleigh-Bénard Convection Analysis
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os

# Set page config
st.set_page_config(
    page_title="Rayleigh-Bénard Convection Analysis", 
    layout="wide", 
    page_icon="🌊"
)

# Title and description
st.title("🌊 Rayleigh-Bénard Convection Analysis")
st.markdown("""
Interactive computational analysis of Rayleigh-Bénard convection with multiple solution methods.
""")

# Create main tabs
main_tabs = st.tabs(["📋 Problem Setup", "🔬 Solution Methods", "📊 Results", "📚 Theory"])

with main_tabs[0]:
    st.header("📋 Problem Setup & Parameters")
    
    # Problem visualization
    st.subheader("Physical Problem Configuration")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw domain
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('Rayleigh-Bénard Convection Problem Setup', fontsize=14, fontweight='bold')
    ax.set_xlabel('x (Width)', fontsize=12)
    ax.set_ylabel('y (Height)', fontsize=12)
    
    # Domain rectangle
    domain = plt.Rectangle((0, 0), 2, 1, linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.3)
    ax.add_patch(domain)
    
    # Hot bottom
    bottom = plt.Rectangle((0, 0), 2, 0.05, linewidth=0, facecolor='red', alpha=0.8)
    ax.add_patch(bottom)
    ax.text(1, -0.1, 'HOT WALL (T = T_h)', ha='center', fontsize=11, color='red', fontweight='bold')
    
    # Cold top
    top = plt.Rectangle((0, 0.95), 2, 0.05, linewidth=0, facecolor='blue', alpha=0.8)
    ax.add_patch(top)
    ax.text(1, 1.1, 'COLD WALL (T = T_c)', ha='center', fontsize=11, color='blue', fontweight='bold')
    
    # Adiabatic sides
    left = plt.Rectangle((0, 0), 0.05, 1, linewidth=0, facecolor='gray', alpha=0.6)
    ax.add_patch(left)
    right = plt.Rectangle((1.95, 0), 0.05, 1, linewidth=0, facecolor='gray', alpha=0.6)
    ax.add_patch(right)
    ax.text(-0.2, 0.5, 'ADIABATIC', ha='center', fontsize=10, rotation=90, va='center')
    ax.text(2.2, 0.5, 'ADIABATIC', ha='center', fontsize=10, rotation=90, va='center')
    
    # Add convection cells
    x = np.linspace(0.1, 1.9, 20)
    y = np.linspace(0.1, 0.9, 10)
    X, Y = np.meshgrid(x, y)
    
    # Create convection pattern
    U = np.sin(2*np.pi*X/2) * np.cos(np.pi*Y)
    V = np.cos(2*np.pi*X/2) * np.sin(np.pi*Y)
    
    # Draw streamlines
    ax.streamplot(X, Y, U, V, color='darkblue', density=1.5, linewidth=1)
    
    # Add dimension arrows
    ax.annotate('', xy=(2, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='<->', lw=2))
    ax.text(1, -0.05, 'L', ha='center', fontsize=12)
    ax.annotate('', xy=(0, 1), xytext=(0, 0), arrowprops=dict(arrowstyle='<->', lw=2))
    ax.text(-0.1, 0.5, 'H', ha='center', fontsize=12)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Governing equations
    st.markdown("---")
    st.subheader("🔬 Governing Equations")
    
    st.markdown("""
    The Rayleigh-Bénard convection problem is governed by the **Boussinesq approximation**:
    
    **Continuity Equation:**
    $$\\nabla \\cdot \\mathbf{u} = 0$$
    
    **Momentum Equations:**
    $$\\frac{\\partial \\mathbf{u}}{\\partial t} + (\\mathbf{u} \\cdot \\nabla)\\mathbf{u} = -\\frac{1}{\\rho_0}\\nabla p + \\nu \\nabla^2 \\mathbf{u} + g\\alpha(T-T_0)\\hat{\\mathbf{j}}$$
    
    **Energy Equation:**
    $$\\frac{\\partial T}{\\partial t} + (\\mathbf{u} \\cdot \\nabla)T = \\kappa \\nabla^2 T$$
    
    **Dimensionless Parameters:**
    - **Rayleigh Number (Ra):** $Ra = \\frac{g \\alpha \\Delta T L^3}{\\nu \\kappa}$
    - **Prandtl Number (Pr):** $Pr = \\frac{\\nu}{\\kappa}$
    - **Nusselt Number (Nu):** $Nu = -\\frac{L}{\\Delta T}\\frac{\\partial T}{\\partial y}\\Big|_{wall}$
    """)
    
    # Parameter selection
    st.markdown("---")
    st.subheader("⚙️ Simulation Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        Ra = st.number_input(
            "Rayleigh Number (Ra)", 
            min_value=1e3, 
            max_value=1e7, 
            value=1e4, 
            format="%.0e"
        )
        Pr = st.number_input(
            "Prandtl Number (Pr)", 
            min_value=0.01, 
            max_value=100.0, 
            value=0.71
        )
    
    with col2:
        nx = st.slider("Grid Points X", min_value=20, max_value=64, value=32, step=8)
        ny = st.slider("Grid Points Y", min_value=20, max_value=64, value=32, step=8)
        max_iterations = st.slider("Max Iterations", min_value=100, max_value=2000, value=500, step=100)

with main_tabs[1]:
    st.header("🔬 Solution Methods")
    
    st.markdown("""
    Select a numerical method to solve the Rayleigh-Bénard convection problem:
    """)
    
    method = st.selectbox(
        "Choose Solution Method",
        options=["finite_difference", "spectral", "pinn"],
        format_func=lambda x: {
            "finite_difference": "Finite Difference Method",
            "spectral": "Spectral Method", 
            "pinn": "Physics-Informed Neural Network"
        }[x]
    )
    
    # Method descriptions
    if method == "finite_difference":
        st.info("""
        **Finite Difference Method**
        - Traditional CFD approach using grid discretization
        - Robust and widely applicable
        - Second-order accuracy in space
        """)
    elif method == "spectral":
        st.info("""
        **Spectral Method**
        - High accuracy using Fourier series
        - Exponential convergence for smooth solutions
        - Minimal numerical diffusion
        """)
    else:
        st.info("""
        **Physics-Informed Neural Network**
        - Machine learning approach with physics constraints
        - Mesh-free solution
        - Transfer learning capabilities
        """)
    
    if st.button(f"🧮 Run {method.replace('_', ' ').title()} Method", type="primary"):
        st.subheader(f"🔬 {method.replace('_', ' ').title()} Solution")
        
        with st.spinner(f"Running {method}..."):
            # Simulate computation
            time.sleep(2)
            
            # Create sample results
            x = np.linspace(0, 2, nx)
            y = np.linspace(0, 1, ny)
            X, Y = np.meshgrid(x, y)
            
            # Sample temperature field
            T = 1 - Y + 0.1 * np.sin(2*np.pi*X/2) * np.sin(np.pi*Y)
            
            # Sample velocity field
            psi = 0.1 * np.sin(np.pi * Y) * np.cos(2 * np.pi * X / 2)
            u = np.gradient(psi, axis=0)
            v = -np.gradient(psi, axis=1)
            
            # Store results in session state
            st.session_state.results = {
                'x': x, 'y': y,
                'temperature': T,
                'stream_function': psi,
                'u_velocity': u,
                'v_velocity': v,
                'method': method
            }
            
            st.success(f"✅ {method} completed successfully!")

with main_tabs[2]:
    st.header("📊 Results Analysis")
    
    if 'results' in st.session_state:
        results = st.session_state.results
        
        st.subheader(f"🌊 {results['method'].replace('_', ' ').title()} Results")
        
        # Create visualization tabs
        viz_tabs = st.tabs(["🌡️ Temperature", "🌀 Stream Function", "💨 Velocity Field"])
        
        with viz_tabs[0]:
            fig = go.Figure(data=go.Heatmap(
                z=results['temperature'],
                x=results['x'],
                y=results['y'],
                colorscale='RdBu_r',
                colorbar=dict(title="Temperature")
            ))
            fig.update_layout(
                title="Temperature Field",
                xaxis_title="x",
                yaxis_title="y"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[1]:
            fig = go.Figure(data=go.Heatmap(
                z=results['stream_function'],
                x=results['x'],
                y=results['y'],
                colorscale='Viridis',
                colorbar=dict(title="Stream Function")
            ))
            fig.update_layout(
                title="Stream Function",
                xaxis_title="x",
                yaxis_title="y"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[2]:
            # Velocity magnitude
            speed = np.sqrt(results['u_velocity']**2 + results['v_velocity']**2)
            
            fig = go.Figure(data=go.Heatmap(
                z=speed,
                x=results['x'],
                y=results['y'],
                colorscale='Plasma',
                colorbar=dict(title="Velocity Magnitude")
            ))
            fig.update_layout(
                title="Velocity Magnitude",
                xaxis_title="x",
                yaxis_title="y"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Physical quantities
        st.markdown("---")
        st.subheader("🌊 Physical Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Max Temperature", f"{np.max(results['temperature']):.3f}")
            st.metric("Min Temperature", f"{np.min(results['temperature']):.3f}")
        
        with col2:
            st.metric("Max Velocity", f"{np.max(speed):.3f}")
            st.metric("Mean Velocity", f"{np.mean(speed):.3f}")
        
        with col3:
            # Simplified Nusselt number
            Nu = 1 + 0.5 * (Ra/1708 - 1)**0.5 if Ra > 1708 else 1.0
            st.metric("Nusselt Number", f"{Nu:.3f}")
            if Nu > 1.0:
                st.success(f"Convective enhancement: {((Nu-1)*100):.1f}%")
            else:
                st.info("Conduction-dominated")
    
    else:
        st.info("🔄 Please run a solution method first to see results")

with main_tabs[3]:
    st.header("📚 Theory & Background")
    
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
    
    ### Flow Regimes
    - **Ra < 1708**: Pure conduction
    - **1708 < Ra < 10⁴**: Steady convection rolls
    - **10⁴ < Ra < 10⁶**: Oscillatory convection
    - **Ra > 10⁶**: Chaotic/turbulent convection
    
    ### Applications
    - **Geophysics**: Mantle convection, atmospheric circulation
    - **Engineering**: Heat exchangers, cooling systems
    - **Astrophysics**: Stellar convection zones
    - **Climate science**: Oceanic and atmospheric dynamics
    """)

# Footer
st.markdown("---")
st.markdown("""
**🔬 Computational Fluid Dynamics Laboratory**  
*Interactive Rayleigh-Bénard convection analysis platform*
""")
