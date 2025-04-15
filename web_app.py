import streamlit as st
import pandas as pd
import numpy as np
import wntr
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from io import StringIO
import tempfile

st.set_page_config(page_title="EPANET Calibration Tool", layout="wide")
st.title("EPANET Calibration + INP Update Tool (Web Edition)")

# --- Load Files ---
inp_file = st.file_uploader("Step 1: Upload EPANET .inp file", type=["inp"])
obs_file = st.file_uploader("Upload Observed Data (CSV)", type=["csv"])

# --- Global State ---
st.session_state.setdefault("calibrated_df", None)
st.session_state.setdefault("rmse_results", {})
st.session_state.setdefault("last_wn", None)

# --- Run Calibration ---
def run_calibration(wn, wn_path, obs_data):
    bounds = [(75, 150)] * len(wn.link_name_list)

    from functools import partial
    def objective(params):
        return objective_wrapper(params, wn_path, wn.link_name_list, obs_data)

    result = differential_evolution(objective, bounds, maxiter=20, workers=1)
    best_params = result.x
    local_result = minimize(objective, best_params, method='L-BFGS-B', bounds=bounds)
    refined_params = local_result.x

    for i, link_name in enumerate(wn.link_name_list):
        wn.get_link(link_name).roughness = refined_params[i]

    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    return wn, results

# --- Objective Function ---
def objective_wrapper(params, wn_path, link_names, obs_data):
    wn = wntr.network.WaterNetworkModel(wn_path)
    for i, link_name in enumerate(link_names):
        wn.get_link(link_name).roughness = params[i]
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    sim_pressures = results.node.get('pressure', pd.DataFrame())
    sim_flows = results.link.get('flowrate', pd.DataFrame())

    error = 0.0
    for col in obs_data.columns:
        if col in sim_pressures.columns:
            sim_series = sim_pressures[col]
        elif col in sim_flows.columns:
            sim_series = sim_flows[col] * 1000
        else:
            continue
        aligned = sim_series.reindex(obs_data.index).interpolate()
        relative_error = ((obs_data[col] - aligned) / obs_data[col]).dropna()
        error += np.nansum(relative_error**2)
    return error

# --- Compute RMSE ---
def compute_rmse(obs_data, results):
    sim_pressures = results.node.get('pressure', pd.DataFrame())
    sim_flows = results.link.get('flowrate', pd.DataFrame()) * 1000
    rmse_results = {}
    error_messages = []

    for col in obs_data.columns:
        if col in sim_pressures.columns:
            sim_series = sim_pressures[col]
            unit = "m"
        elif col in sim_flows.columns:
            sim_series = sim_flows[col]
            unit = "L/s"
        else:
            continue

        aligned = sim_series.reindex(obs_data.index).interpolate()
        diff = obs_data[col] - aligned
        if diff.dropna().empty:
            rmse = np.nan
        else:
            rmse = np.sqrt(np.nanmean(diff**2))
            rmse_results[col] = rmse
        error_messages.append(f"RMSE {col}: {rmse:.2f} {unit}")

    return rmse_results, error_messages

# --- Plotting ---
def plot_results(obs_data, results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for col in obs_data.columns:
        if col in results.node.get('pressure', pd.DataFrame()).columns:
            sim_series = results.node['pressure'][col]
            aligned = sim_series.reindex(obs_data.index).interpolate()
            ax1.plot(obs_data.index, obs_data[col], label=f"Obs {col}")
            ax1.plot(obs_data.index, aligned, label=f"Sim {col}")
        elif col in results.link.get('flowrate', pd.DataFrame()).columns:
            sim_series = results.link['flowrate'][col] * 1000
            aligned = sim_series.reindex(obs_data.index).interpolate()
            ax2.plot(obs_data.index, obs_data[col], label=f"Obs {col}")
            ax2.plot(obs_data.index, aligned, label=f"Sim {col}")

    ax1.set_title("Observed vs Simulated Pressure")
    ax1.set_ylabel("Pressure (m)")
    ax1.legend()
    ax2.set_title("Observed vs Simulated Flow")
    ax2.set_ylabel("Flow (L/s)")
    ax2.set_xlabel("Time")
    ax2.legend()

    st.pyplot(fig)

# --- Plot Network ---
def plot_network(wn):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    # Plot the network structure
    wntr.graphics.plot_network(
        wn,
        node_attribute=None,
        link_attribute=None,
        node_size=12,
        link_width=1.0,
        title="Network Map with Initial & Calibrated Roughness Labels",
        ax=ax
    )

    # Annotate each pipe with initial and calibrated roughness
    for link_name, link in wn.links():
        try:
            x = (link.start_node.coordinates[0] + link.end_node.coordinates[0]) / 2
            y = (link.start_node.coordinates[1] + link.end_node.coordinates[1]) / 2
        except:
            continue

        roughness = link.roughness
        init_val = None

        if "calibrated_df" in st.session_state and st.session_state["calibrated_df"] is not None:
            df = st.session_state["calibrated_df"]
            if link_name in df["Pipe"].values:
                init_val = df.loc[df["Pipe"] == link_name, "Calibrated Roughness"].values[0]

        label = f"{roughness:.1f}"
        if init_val is not None and not np.isclose(init_val, roughness):
            label = f"{init_val:.1f} → {roughness:.1f}"

        ax.text(x, y, label, color='blue', fontsize=8, ha='center')

    plt.tight_layout()
    st.pyplot(fig)

    # Export PNG button
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.download_button(

        label="Download Network Map as PNG",
        data=buf,
        file_name="network_map.png",
        mime="image/png"
    )

# --- Main Logic ---
if inp_file and obs_file:
    temp_path = f"temp_model.inp"
    with open(temp_path, "wb") as f:
        f.write(inp_file.getvalue())

    obs_data = pd.read_csv(obs_file, index_col=0)
    wn = wntr.network.WaterNetworkModel(temp_path)

    if st.button("Run Calibration"):
        with st.spinner("Running calibration. Please wait..."):
            wn, results = run_calibration(wn, temp_path, obs_data)
            rmse, messages = compute_rmse(obs_data, results)
            st.session_state["rmse_results"] = rmse
            st.session_state["last_wn"] = wn

            for msg in messages:
                st.write(msg)

            plot_results(obs_data, results)

            calibrated_data = [[link_name, wn.get_link(link_name).roughness] for link_name in wn.link_name_list]
            st.session_state["calibrated_df"] = pd.DataFrame(calibrated_data, columns=["Pipe", "Calibrated Roughness"])

            st.subheader("📌 Network Map (Colored by Roughness)")
            plot_network(wn)

# --- Export Roughness ---
if st.session_state["calibrated_df"] is not None:
    st.download_button("Download Calibrated Roughness CSV", st.session_state["calibrated_df"].to_csv(index=False), file_name="calibrated_roughness.csv")

# --- Step 2: Update INP File ---
st.subheader("Step 2: Update INP File with Calibrated Roughness")
roughness_csv = st.file_uploader("Upload Roughness CSV (Pipe, Calibrated Roughness)", type=["csv"], key="update_csv")
inp_to_update = st.file_uploader("Upload INP File to Update", type=["inp"], key="update_inp")

if roughness_csv and inp_to_update:
    rough_df = pd.read_csv(roughness_csv)
    rough_dict = dict(zip(rough_df['Pipe'], rough_df['Calibrated Roughness']))

    inp_lines = inp_to_update.getvalue().decode("utf-8").splitlines()
    start, end = None, None
    for i, line in enumerate(inp_lines):
        if line.strip().upper() == "[PIPES]":
            start = i
        elif start and line.strip().startswith("[") and line.strip().endswith("]"):
            end = i
            break

    for i in range(start + 2, end):
        parts = inp_lines[i].split()
        if not parts or parts[0].startswith(";"):
            continue
        pipe_id = parts[0]
        if pipe_id in rough_dict:
            parts[5] = f"{rough_dict[pipe_id]:.6f}"
            inp_lines[i] = "\t".join(parts)

    updated_content = "\n".join(inp_lines)
    from io import BytesIO
    st.download_button(
        label="Download Updated INP File",
        data=BytesIO(updated_content.encode("utf-8")),
        file_name="updated_model.inp",
        mime="text/plain"
    )

    
