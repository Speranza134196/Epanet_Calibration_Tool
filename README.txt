EPANET Calibration + INP Update Tool (Web Edition)
Overview
This tool provides a browser-based interface to:
* Calibrate pipe roughness values in an EPANET model using observed data
* Visualize simulation accuracy (RMSE, plots)
* Export calibrated data
* Update an existing INP file with new roughness values
Built with Streamlit, it’s simple, visual, and does not require deep programming knowledge.

Step 1: Calibrate the Network
1. Upload EPANET .inp File
* Format: A valid EPANET model file (e.g., network.inp).
2. Upload Observed Data (CSV)
* Time series of pressure or flow data.
* Format Example:
* Time, Node1, Node2, Pipe5
* 00:00, 45.3, 46.2, 18.7
* 01:00, 44.7, 45.9, 18.1
3. Click "Run Calibration"
* Uses optimization to minimize the difference between observed and simulated values.
* Outputs:
o RMSE values per measurement point
o Plots: Observed vs Simulated
o Network diagram with labels (initial ? calibrated roughness)
4. Download Calibrated Data
* A CSV is generated with Pipe and Calibrated Roughness values.

Step 2: Update an INP File
1. Upload Calibrated Roughness CSV
* Columns: Pipe, Calibrated Roughness
2. Upload Existing .inp File
* The tool scans the [PIPES] section and updates the roughness values.
3. Download Updated .inp
* All changes are saved and downloadable as a new INP file.
Note: Network visualization is not shown in this step to keep the interface clean.

Visual Outputs
* ?? Time Series Plots
o Shows observed vs simulated pressure and flow
* ??? Network Map
o Annotated with roughness values per pipe
o Format: initial ? calibrated
o Downloadable as PNG

Notes
* Assumes pressure in meters (m) and flow in liters/second (L/s)
* Missing data is skipped without breaking the calibration
* INP file must include node and pipe coordinates for plotting

Future Enhancements (Ideas)
* Attribute toggle for plotting (e.g., pressure, flow, roughness)
* Interactive maps using Plotly
* Error reporting per node or pipe

Author
VIASTRATA Systems – For water network calibration, the smooth way ??

