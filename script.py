"""
source .venv/bin/activate.fish
ipython
%load_ext autoreload
%autoreload 2
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
"""

from indica.workflow import JetWorkflow
from indica.workflow import PlotJetWorkflow

workflow = JetWorkflow(config_file="90279.json")

workflow.get_diagnostics()
workflow.fit_diagnostic_profiles()

workflow.invert_sxr()

plot_workflow = PlotJetWorkflow(workflow, default_time=46.17)
plot_workflow.plot_los()
plot_workflow.plot_sxr_los_fit()
plot_workflow.plot_emissivity_midplane_rho(time=45)
