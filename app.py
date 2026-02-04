import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.nn as nn
import time
from scipy import sparse
from scipy.sparse.linalg import spsolve
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Convection Problem Solver", layout="wide", page_icon="🌊")

# Title and description
st.title("🌊 AI-Powered Rayleigh-Bénard Convection Solver")
st.markdown("""
This application solves Rayleigh-Bénard convection problems using both **traditional numerical methods** 
and **Physics-Informed Neural Networks (PINNs)**. Compare solutions, validate ML models, and explore 
different convection regimes!
""")

# Sidebar for parameters
st.sidebar.header("⚙️ Problem Parameters")

# Physical parameters - more conservative defaults
Ra = st.sidebar.number_input("Rayleigh Number (Ra)", min_value=1e3, max_value=1e5, value=5e3, 
                             format="%.0e", help="Ratio of buoyancy to viscous forces")
Pr = st.sidebar.number_input("Prandtl Number (Pr)", min_value=0.1, max_value=10.0, value=0.71,
                             help="Ratio of momentum diffusivity to thermal diffusivity")
aspect_ratio = st.sidebar.slider("Aspect Ratio", min_value=0.5, max_value=4.0, value=2.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("🔧 Numerical Parameters")

# Grid resolution
nx = st.sidebar.slider("Grid Points (X)", min_value=20, max_value=100, value=50, step=10)
ny = st.sidebar.slider("Grid Points (Y)", min_value=20, max_value=100, value=50, step=10)
max_iterations = st.sidebar.slider("Max Iterations", min_value=100, max_value=5000, value=1000, step=100)
tolerance = st.sidebar.number_input("Convergence Tolerance", min_value=1e-8, max_value=1e-3, 
                                    value=1e-5, format="%.0e")

st.sidebar.markdown("---")
st.sidebar.header("🧠 Neural Network Parameters")

# PINN parameters - more conservative defaults
hidden_layers = st.sidebar.slider("Hidden Layers", min_value=2, max_value=6, value=3)
neurons_per_layer = st.sidebar.slider("Neurons per Layer", min_value=20, max_value=80, value=30, step=10)
training_epochs = st.sidebar.slider("Training Epochs", min_value=1000, max_value=10000, value=3000, step=500)
learning_rate = st.sidebar.number_input("Learning Rate", min_value=1e-4, max_value=1e-2, 
                                        value=5e-4, format="%.0e")

# Add helpful tips
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips for stability:**\n- Start with Ra ≤ 1e4\n- Use 30×30 grid for testing\n- Increase gradually")


# ============================================================================
# TRADITIONAL SOLVER - Finite Difference Method
# ============================================================================

class TraditionalSolver:
    """
    Solves 2D Rayleigh-Bénard convection using finite difference method
    with Boussinesq approximation in stream function-vorticity formulation
    """
    
    def __init__(self, nx, ny, Ra, Pr, aspect_ratio, max_iter=1000, tol=1e-5):
        self.nx = nx
        self.ny = ny
        self.Ra = Ra
        self.Pr = Pr
        self.aspect_ratio = aspect_ratio
        self.max_iter = max_iter
        self.tol = tol
        
        # Domain
        self.Lx = aspect_ratio
        self.Ly = 1.0
        self.dx = self.Lx / (nx - 1)
        self.dy = self.Ly / (ny - 1)
        
        # Mesh
        self.x = np.linspace(0, self.Lx, nx)
        self.y = np.linspace(0, self.Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize fields
        self.T = np.zeros((ny, nx))
        self.psi = np.zeros((ny, nx))
        self.omega = np.zeros((ny, nx))
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        
    def set_initial_conditions(self):
        """Set initial temperature field with small perturbation"""
        # Linear temperature profile with perturbation
        for j in range(self.ny):
            self.T[j, :] = 1.0 - self.y[j] + 0.1 * np.sin(np.pi * self.x / self.Lx) * np.sin(np.pi * self.y[j])
        
    def apply_boundary_conditions(self):
        """Apply boundary conditions: hot bottom, cold top, insulated sides"""
        # Temperature BCs
        self.T[0, :] = 1.0  # Hot bottom
        self.T[-1, :] = 0.0  # Cold top
        
        # Stream function BCs (no-slip walls)
        self.psi[:, 0] = 0
        self.psi[:, -1] = 0
        self.psi[0, :] = 0
        self.psi[-1, :] = 0
        
    def solve_poisson(self, source, bc_value=0):
        """Solve Poisson equation using sparse matrix solver"""
        n = (self.nx - 2) * (self.ny - 2)
        
        # Build sparse matrix for 2D Poisson equation
        diagonals = [
            -2 * (1/self.dx**2 + 1/self.dy**2) * np.ones(n),  # main diagonal
            (1/self.dx**2) * np.ones(n-1),  # off-diagonal
            (1/self.dx**2) * np.ones(n-1),
            (1/self.dy**2) * np.ones(n-(self.nx-2)),
            (1/self.dy**2) * np.ones(n-(self.nx-2))
        ]
        
        A = sparse.diags(diagonals, [0, -1, 1, -(self.nx-2), (self.nx-2)], format='csr')
        
        # RHS with boundary conditions
        b = source[1:-1, 1:-1].flatten()
        
        # Solve
        solution = spsolve(A, b)
        
        # Reshape and apply boundary conditions
        result = np.zeros((self.ny, self.nx))
        result[1:-1, 1:-1] = solution.reshape((self.ny-2, self.nx-2))
        result[:, 0] = bc_value
        result[:, -1] = bc_value
        result[0, :] = bc_value
        result[-1, :] = bc_value
        
        return result
    
    def compute_velocities(self):
        """Compute velocities from stream function"""
        # u = d(psi)/dy
        self.u[1:-1, 1:-1] = (self.psi[2:, 1:-1] - self.psi[:-2, 1:-1]) / (2 * self.dy)
        
        # v = -d(psi)/dx
        self.v[1:-1, 1:-1] = -(self.psi[1:-1, 2:] - self.psi[1:-1, :-2]) / (2 * self.dx)
        
    def solve(self):
        """Main solver loop"""
        self.set_initial_conditions()
        
        # Time stepping parameters - very conservative for stability
        dt = min(0.0001, 0.01 / max(self.Ra, 1e4))  # Much smaller dt
        
        # Add under-relaxation for stability
        alpha = 0.1  # Under-relaxation factor
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for iteration in range(self.max_iter):
            T_old = self.T.copy()
            
            # Update boundary conditions
            self.apply_boundary_conditions()
            
            # Compute vorticity from temperature using buoyancy term
            # omega = Ra * Pr * dT/dx
            dTdx = (self.T[1:-1, 2:] - self.T[1:-1, :-2]) / (2 * self.dx)
            self.omega[1:-1, 1:-1] = self.Ra * self.Pr * dTdx
            
            # Limit vorticity to prevent overflow
            max_omega = 1e6
            self.omega = np.clip(self.omega, -max_omega, max_omega)
            
            # Solve for stream function from vorticity
            # ∇²ψ = -ω
            self.psi = self.solve_poisson(-self.omega)
            
            # Compute velocities
            self.compute_velocities()
            
            # Update temperature using advection-diffusion with under-relaxation
            # dT/dt + u*dT/dx + v*dT/dy = ∇²T
            dTdx = np.zeros_like(self.T)
            dTdy = np.zeros_like(self.T)
            d2Tdx2 = np.zeros_like(self.T)
            d2Tdy2 = np.zeros_like(self.T)
            
            dTdx[1:-1, 1:-1] = (self.T[1:-1, 2:] - self.T[1:-1, :-2]) / (2 * self.dx)
            dTdy[1:-1, 1:-1] = (self.T[2:, 1:-1] - self.T[:-2, 1:-1]) / (2 * self.dy)
            
            d2Tdx2[1:-1, 1:-1] = (self.T[1:-1, 2:] - 2*self.T[1:-1, 1:-1] + self.T[1:-1, :-2]) / self.dx**2
            d2Tdy2[1:-1, 1:-1] = (self.T[2:, 1:-1] - 2*self.T[1:-1, 1:-1] + self.T[:-2, 1:-1]) / self.dy**2
            
            # Limit velocities to prevent instability
            max_vel = 10.0
            self.u = np.clip(self.u, -max_vel, max_vel)
            self.v = np.clip(self.v, -max_vel, max_vel)
            
            # Compute temperature update with under-relaxation
            dT_dt = d2Tdx2[1:-1, 1:-1] + d2Tdy2[1:-1, 1:-1] - \
                    self.u[1:-1, 1:-1] * dTdx[1:-1, 1:-1] - \
                    self.v[1:-1, 1:-1] * dTdy[1:-1, 1:-1]
            
            # Apply under-relaxation
            self.T[1:-1, 1:-1] += alpha * dt * dT_dt
            
            # Check convergence and stability (only after minimum iterations)
            error = np.max(np.abs(self.T - T_old))
            
            # Check for NaN or infinity
            if np.any(np.isnan(self.T)) or np.any(np.isinf(self.T)):
                status_text.text("❌ Simulation became unstable! Try smaller Ra or larger grid.")
                progress_bar.progress(1.0)
                return None
            
            if iteration % 100 == 0:
                progress_bar.progress(min(iteration / self.max_iter, 1.0))
                status_text.text(f"Iteration {iteration}/{self.max_iter}, Error: {error:.2e}")
            
            # Only check convergence after minimum iterations to avoid false convergence
            if iteration > 50 and error < self.tol:
                status_text.text(f"✅ Converged in {iteration} iterations! Error: {error:.2e}")
                progress_bar.progress(1.0)
                break
        
        return {
            'temperature': self.T,
            'stream_function': self.psi,
            'u_velocity': self.u,
            'v_velocity': self.v,
            'x': self.x,
            'y': self.y,
            'X': self.X,
            'Y': self.Y
        }


# ============================================================================
# PINN SOLVER - Physics-Informed Neural Network
# ============================================================================

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for Rayleigh-Bénard convection
    """
    
    def __init__(self, layers):
        super(PINN, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        # Initialize weights
        for m in self.layers:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x, y):
        """Forward pass through the network"""
        inputs = torch.cat([x, y], dim=1)
        
        for i, layer in enumerate(self.layers[:-1]):
            inputs = torch.tanh(layer(inputs))
        
        # Output layer (no activation)
        outputs = self.layers[-1](inputs)
        
        # outputs: [T, psi, u, v]
        return outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3], outputs[:, 3:4]


class PINNSolver:
    """
    Solver for Rayleigh-Bénard convection using PINN
    """
    
    def __init__(self, nx, ny, Ra, Pr, aspect_ratio, layers, epochs=5000, lr=1e-3):
        self.nx = nx
        self.ny = ny
        self.Ra = Ra
        self.Pr = Pr
        self.aspect_ratio = aspect_ratio
        self.epochs = epochs
        self.lr = lr
        
        # Domain
        self.Lx = aspect_ratio
        self.Ly = 1.0
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Network
        self.model = PINN(layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Generate training points
        self.generate_training_points()
        
    def generate_training_points(self):
        """Generate collocation points for training"""
        # Interior points
        n_interior = 2000
        x_interior = torch.rand(n_interior, 1) * self.Lx
        y_interior = torch.rand(n_interior, 1) * self.Ly
        
        # Boundary points
        n_boundary = 500
        
        # Bottom boundary (y=0, hot)
        x_bottom = torch.rand(n_boundary, 1) * self.Lx
        y_bottom = torch.zeros(n_boundary, 1)
        
        # Top boundary (y=1, cold)
        x_top = torch.rand(n_boundary, 1) * self.Lx
        y_top = torch.ones(n_boundary, 1)
        
        # Left boundary
        x_left = torch.zeros(n_boundary, 1)
        y_left = torch.rand(n_boundary, 1) * self.Ly
        
        # Right boundary
        x_right = torch.ones(n_boundary, 1) * self.Lx
        y_right = torch.rand(n_boundary, 1) * self.Ly
        
        # Move to device
        self.x_interior = x_interior.requires_grad_(True).to(self.device)
        self.y_interior = y_interior.requires_grad_(True).to(self.device)
        
        self.x_bottom = x_bottom.to(self.device)
        self.y_bottom = y_bottom.to(self.device)
        self.x_top = x_top.to(self.device)
        self.y_top = y_top.to(self.device)
        self.x_left = x_left.to(self.device)
        self.y_left = y_left.to(self.device)
        self.x_right = x_right.to(self.device)
        self.y_right = y_right.to(self.device)
    
    def physics_loss(self, x, y):
        """Compute physics-informed loss"""
        T, psi, u, v = self.model(x, y)
        
        # Compute gradients
        T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]
        T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y), create_graph=True)[0]
        
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        
        psi_x = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        psi_y = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        
        # Energy equation: dT/dt + u*dT/dx + v*dT/dy = ∇²T (steady state: dT/dt = 0)
        energy_residual = u * T_x + v * T_y - (T_xx + T_yy)
        
        # Stream function relations: u = dpsi/dy, v = -dpsi/dx
        continuity_u = u - psi_y
        continuity_v = v + psi_x
        
        # Incompressibility: du/dx + dv/dy = 0
        incompressibility = u_x + v_y
        
        # Total physics loss
        loss_physics = torch.mean(energy_residual**2) + \
                      torch.mean(continuity_u**2) + \
                      torch.mean(continuity_v**2) + \
                      torch.mean(incompressibility**2)
        
        return loss_physics
    
    def boundary_loss(self):
        """Compute boundary condition loss"""
        # Bottom: T = 1 (hot)
        T_bottom, psi_bottom, u_bottom, v_bottom = self.model(self.x_bottom, self.y_bottom)
        loss_bottom = torch.mean((T_bottom - 1.0)**2) + torch.mean(u_bottom**2) + torch.mean(v_bottom**2)
        
        # Top: T = 0 (cold)
        T_top, psi_top, u_top, v_top = self.model(self.x_top, self.y_top)
        loss_top = torch.mean(T_top**2) + torch.mean(u_top**2) + torch.mean(v_top**2)
        
        # Left and right: no-slip (u=v=0), adiabatic (dT/dx=0)
        T_left, psi_left, u_left, v_left = self.model(self.x_left, self.y_left)
        T_right, psi_right, u_right, v_right = self.model(self.x_right, self.y_right)
        
        loss_sides = torch.mean(u_left**2) + torch.mean(v_left**2) + \
                     torch.mean(u_right**2) + torch.mean(v_right**2)
        
        return loss_bottom + loss_top + loss_sides
    
    def train(self):
        """Train the PINN"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_history = []
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            
            # Compute losses
            loss_phys = self.physics_loss(self.x_interior, self.y_interior)
            loss_bc = self.boundary_loss()
            
            # Total loss
            loss = loss_phys + 10.0 * loss_bc  # Weight BC loss higher
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                status_text.text("❌ Training became unstable! Try smaller learning rate.")
                progress_bar.progress(1.0)
                return None
            
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            
            loss_history.append(loss.item())
            
            # Early stopping with patience
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter > 1000:  # Stop if no improvement for 1000 epochs
                status_text.text(f"✅ Early stopping at epoch {epoch}! Best loss: {best_loss:.4e}")
                progress_bar.progress(1.0)
                break
            
            if epoch % 100 == 0:
                progress_bar.progress(min(epoch / self.epochs, 1.0))
                status_text.text(f"Epoch {epoch}/{self.epochs}, Loss: {loss.item():.4e}")
        
        status_text.text(f"✅ Training complete! Final loss: {loss.item():.4e}")
        progress_bar.progress(1.0)
        
        return loss_history
    
    def predict(self, x_grid, y_grid):
        """Make predictions on a grid"""
        self.model.eval()
        
        x_flat = torch.tensor(x_grid.flatten(), dtype=torch.float32).reshape(-1, 1).to(self.device)
        y_flat = torch.tensor(y_grid.flatten(), dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        with torch.no_grad():
            T, psi, u, v = self.model(x_flat, y_flat)
        
        T = T.cpu().numpy().reshape(x_grid.shape)
        psi = psi.cpu().numpy().reshape(x_grid.shape)
        u = u.cpu().numpy().reshape(x_grid.shape)
        v = v.cpu().numpy().reshape(x_grid.shape)
        
        return T, psi, u, v


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_comparison(traditional_results, pinn_results, field_name):
    """Create side-by-side comparison plots"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Traditional Solver', 'PINN Solver', 'Absolute Difference'),
        horizontal_spacing=0.1
    )
    
    # Get field data
    if field_name == 'Temperature':
        trad_field = traditional_results['temperature']
        pinn_field = pinn_results['temperature']
        colorscale = 'RdBu_r'
    elif field_name == 'Stream Function':
        trad_field = traditional_results['stream_function']
        pinn_field = pinn_results['stream_function']
        colorscale = 'Viridis'
    elif field_name == 'U Velocity':
        trad_field = traditional_results['u_velocity']
        pinn_field = pinn_results['u_velocity']
        colorscale = 'RdBu_r'
    else:  # V Velocity
        trad_field = traditional_results['v_velocity']
        pinn_field = pinn_results['v_velocity']
        colorscale = 'RdBu_r'
    
    X = traditional_results['X']
    Y = traditional_results['Y']
    
    # Traditional solver
    fig.add_trace(
        go.Heatmap(z=trad_field, x=X[0], y=Y[:, 0], colorscale=colorscale, 
                   name='Traditional', showscale=False),
        row=1, col=1
    )
    
    # PINN solver
    fig.add_trace(
        go.Heatmap(z=pinn_field, x=X[0], y=Y[:, 0], colorscale=colorscale,
                   name='PINN', showscale=False),
        row=1, col=2
    )
    
    # Difference
    diff = np.abs(trad_field - pinn_field)
    fig.add_trace(
        go.Heatmap(z=diff, x=X[0], y=Y[:, 0], colorscale='Hot',
                   name='Difference', colorbar=dict(x=1.15)),
        row=1, col=3
    )
    
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=2)
    fig.update_xaxes(title_text="x", row=1, col=3)
    fig.update_yaxes(title_text="y", row=1, col=1)
    
    fig.update_layout(height=400, title_text=f"{field_name} Comparison")
    
    return fig


def compute_metrics(traditional_results, pinn_results):
    """Compute error metrics between traditional and PINN solutions"""
    metrics = {}
    
    for field_name, trad_key, pinn_key in [
        ('Temperature', 'temperature', 'temperature'),
        ('U Velocity', 'u_velocity', 'u_velocity'),
        ('V Velocity', 'v_velocity', 'v_velocity')
    ]:
        trad = traditional_results[trad_key]
        pinn = pinn_results[pinn_key]
        
        # L2 error
        l2_error = np.sqrt(np.mean((trad - pinn)**2))
        
        # Relative L2 error
        rel_l2_error = l2_error / (np.sqrt(np.mean(trad**2)) + 1e-10)
        
        # Max absolute error
        max_error = np.max(np.abs(trad - pinn))
        
        metrics[field_name] = {
            'L2 Error': l2_error,
            'Relative L2 Error': rel_l2_error,
            'Max Error': max_error
        }
    
    return metrics


# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔬 Solve & Compare", "📊 Analysis", "📚 About"])

with tab1:
    st.header("Solver Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        solve_traditional = st.button("🧮 Run Traditional Solver", type="primary", use_container_width=True)
    
    with col2:
        solve_pinn = st.button("🧠 Run PINN Solver", type="primary", use_container_width=True)
    
    # Initialize session state
    if 'traditional_results' not in st.session_state:
        st.session_state.traditional_results = None
    if 'pinn_results' not in st.session_state:
        st.session_state.pinn_results = None
    if 'traditional_time' not in st.session_state:
        st.session_state.traditional_time = None
    if 'pinn_time' not in st.session_state:
        st.session_state.pinn_time = None
    
    # Run traditional solver
    if solve_traditional:
        st.subheader("Traditional Finite Difference Solver")
        start_time = time.time()
        
        solver = TraditionalSolver(nx, ny, Ra, Pr, aspect_ratio, max_iterations, tolerance)
        results = solver.solve()
        
        if results is None:
            st.error("❌ Traditional solver failed! Try reducing Rayleigh number or increasing grid resolution.")
        else:
            st.session_state.traditional_results = results
            st.session_state.traditional_time = time.time() - start_time
            st.success(f"✅ Traditional solver completed in {st.session_state.traditional_time:.2f} seconds")
            
            # Show individual traditional solver results
            if st.session_state.traditional_results is not None:
                st.markdown("---")
                st.subheader("📈 Traditional Solver Results")
                
                # Quick visualization
                field_to_show = st.selectbox(
                    "Select field to view:",
                    ['Temperature', 'Stream Function', 'U Velocity', 'V Velocity'],
                    key="trad_field"
                )
                
                # Create simple plot
                if field_to_show == 'Temperature':
                    field_data = st.session_state.traditional_results['temperature']
                elif field_to_show == 'Stream Function':
                    field_data = st.session_state.traditional_results['stream_function']
                elif field_to_show == 'U Velocity':
                    field_data = st.session_state.traditional_results['u_velocity']
                else:
                    field_data = st.session_state.traditional_results['v_velocity']
                
                fig = go.Figure(data=go.Heatmap(
                    z=field_data,
                    x=st.session_state.traditional_results['x'],
                    y=st.session_state.traditional_results['y'],
                    colorscale='RdBu_r'
                ))
                fig.update_layout(title=f"{field_to_show} - Traditional Solver", height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Run PINN solver
    if solve_pinn:
        st.subheader("Physics-Informed Neural Network Solver")
        start_time = time.time()
        
        # Define network architecture
        layers = [2] + [neurons_per_layer] * hidden_layers + [4]  # Input: (x,y), Output: (T, psi, u, v)
        
        pinn_solver = PINNSolver(nx, ny, Ra, Pr, aspect_ratio, layers, training_epochs, learning_rate)
        loss_history = pinn_solver.train()
        
        if loss_history is None:
            st.error("❌ PINN training failed! Try reducing learning rate or changing network architecture.")
        else:
            # Generate prediction grid
            x_grid = np.linspace(0, aspect_ratio, nx)
            y_grid = np.linspace(0, 1.0, ny)
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            
            T_pred, psi_pred, u_pred, v_pred = pinn_solver.predict(X_grid, Y_grid)
            
            st.session_state.pinn_results = {
                'temperature': T_pred,
                'stream_function': psi_pred,
                'u_velocity': u_pred,
                'v_velocity': v_pred,
                'X': X_grid,
                'Y': Y_grid,
                'loss_history': loss_history
            }
            st.session_state.pinn_time = time.time() - start_time
            st.success(f"✅ PINN solver completed in {st.session_state.pinn_time:.2f} seconds")
            
            # Show individual PINN solver results
            if st.session_state.pinn_results is not None:
                st.markdown("---")
                st.subheader("🧠 PINN Solver Results")
                
                # Quick visualization
                field_to_show = st.selectbox(
                    "Select field to view:",
                    ['Temperature', 'Stream Function', 'U Velocity', 'V Velocity'],
                    key="pinn_field"
                )
                
                # Create simple plot
                if field_to_show == 'Temperature':
                    field_data = st.session_state.pinn_results['temperature']
                elif field_to_show == 'Stream Function':
                    field_data = st.session_state.pinn_results['stream_function']
                elif field_to_show == 'U Velocity':
                    field_data = st.session_state.pinn_results['u_velocity']
                else:
                    field_data = st.session_state.pinn_results['v_velocity']
                
                fig = go.Figure(data=go.Heatmap(
                    z=field_data,
                    x=st.session_state.pinn_results['X'][0],
                    y=st.session_state.pinn_results['Y'][:, 0],
                    colorscale='RdBu_r'
                ))
                fig.update_layout(title=f"{field_to_show} - PINN Solver", height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Display results if both are available
    if st.session_state.traditional_results is not None and st.session_state.pinn_results is not None:
        st.markdown("---")
        st.header("📊 Results Comparison")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Traditional Solver Time", f"{st.session_state.traditional_time:.2f}s")
        with col2:
            st.metric("PINN Solver Time", f"{st.session_state.pinn_time:.2f}s")
        with col3:
            speedup = st.session_state.traditional_time / st.session_state.pinn_time
            st.metric("Speedup", f"{speedup:.2f}x", 
                     delta="PINN faster" if speedup > 1 else "Traditional faster")
        
        # Field selector
        field_to_plot = st.selectbox(
            "Select field to compare:",
            ['Temperature', 'Stream Function', 'U Velocity', 'V Velocity']
        )
        
        # Plot comparison
        fig = plot_comparison(st.session_state.traditional_results, 
                            st.session_state.pinn_results, 
                            field_to_plot)
        st.plotly_chart(fig, use_container_width=True)
        
        # Error metrics
        st.subheader("📈 Error Metrics")
        metrics = compute_metrics(st.session_state.traditional_results, st.session_state.pinn_results)
        
        metric_cols = st.columns(len(metrics))
        for idx, (field, values) in enumerate(metrics.items()):
            with metric_cols[idx]:
                st.markdown(f"**{field}**")
                st.write(f"L2 Error: {values['L2 Error']:.4e}")
                st.write(f"Rel. L2: {values['Relative L2 Error']:.4e}")
                st.write(f"Max Error: {values['Max Error']:.4e}")

with tab2:
    st.header("Detailed Analysis")
    
    if st.session_state.pinn_results is not None and 'loss_history' in st.session_state.pinn_results:
        st.subheader("PINN Training Loss History")
        
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=st.session_state.pinn_results['loss_history'],
            mode='lines',
            name='Training Loss'
        ))
        fig_loss.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Loss",
            yaxis_type="log",
            height=400
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    
    if st.session_state.traditional_results is not None:
        st.subheader("Nusselt Number Analysis")
        st.info("Nusselt number quantifies convective heat transfer. Coming soon!")

with tab3:
    st.header("About This Application")
    
    st.markdown("""
    ### 🌊 Rayleigh-Bénard Convection
    
    This application solves the classic Rayleigh-Bénard convection problem, where a fluid layer is heated 
    from below and cooled from above, leading to convective motion.
    
    **Key Parameters:**
    - **Rayleigh Number (Ra)**: Ratio of buoyancy to viscous forces. Higher Ra → stronger convection
    - **Prandtl Number (Pr)**: Ratio of momentum to thermal diffusivity
    - **Aspect Ratio**: Width to height ratio of the domain
    
    ### 🔬 Solution Methods
    
    #### Traditional Solver
    Uses finite difference method with stream function-vorticity formulation. Robust and well-established.
    
    #### PINN Solver
    Physics-Informed Neural Network that learns the solution by minimizing both boundary conditions 
    and governing equations (Navier-Stokes + energy equation).
    
    ### 📊 Validation
    
    The ML model is validated against the traditional solver using:
    - L2 error norms
    - Maximum absolute error
    - Visual comparison of fields
    - Computational performance
    
    ### 🚀 Future Enhancements
    - Multiple convection problems (natural, mixed, forced)
    - 3D convection
    - Turbulent regimes
    - Transfer learning capabilities
    - Experimental data validation
    
    ### 📝 Citation
    If you use this tool in your research, please cite our work (publication pending).
    
    **Created with ❤️ for the fluid mechanics and ML community**
    """)
    
    st.markdown("---")
    st.markdown("**Tech Stack:** Streamlit • PyTorch • NumPy • SciPy • Plotly")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🌊 Convection Problem Solver | Open Source on GitHub | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
