

## Description 

Code and minimal dataset for "Estimation of Cochlear Frequency Selectivity Using a Convolution Model of Forward-Masked Compound Action Potentials". F. Deloche, S. Parida, A. Sivaprakasam, M.G. Heinz, *JARO* (2024). https://doi.org/10.1007/s10162-023-00922-1

Zenodo badge: [![DOI](https://zenodo.org/badge/476437527.svg)](https://zenodo.org/badge/latestdoi/476437527)



Averaged data
-------

We provide the average CAP data associated with each masking condition for each of the four experiments presented in the paper in folder `averaged_data`. See the notebooks in `utils/Data` for how to display data. More specifically, there are 10 averaged response signals in each 'picture file', corresponding to the 10 repetitions of stimulus by block (odd/even indices alternate the polarity of clicks).

## Lab notes 

Some of these notes are specific to the data pipeline in the Heinz lab, but it describes nevertheless the procedure from raw data processing to analysis of results.

### Steps for fitting exper. data

#### Pre- fitting procedure

* The experimental raw Data (.mat files) is found in `Data/[expe-name]/`  (averaged data for github repo is in folder `averaged_data`). picture files: format similar to CAP data in the Heinz lab (they also provide information on the  masker).

  * for the raw data, these variables are used to select the pictures used for analysis (see notebooks in `utils/Data`) : `data_folder `, `begin_ind`, `end_ind` (start/end pic numbers; some experiments also use a list of pic numbers to ignore `pic_numbers_ignore`)

* In main folder: a file `fit_data_list_maskers.py` provides the list of maskers. For each CF:

	* `ntch`: notched noise maskers with diff atten for the notch (I/O function)
	* `vbw` various bandwidths (notch widths, estim of 10dB bandwidth)
	* `vfreq` various frequencies (for the notch) around CF (more maskers for the estim of frequency weights)

* In main folder: a file `fit_data_common.py` is used to process the data before analysis. Contains code/info for setting reference intensity, pre-processing of CAP responses, narrow-band analysis and estimation of latencies. Parameters include:

	* `data_folder`

	* `I0` based on NEL output (during exper., this is displayed in NEL: 'max output should be 105 dB for RMS=1', number can change) + masker atten

	* `begin_ind`, `end_ind` , `pic_numbers_ignore`

	* `t0`, `t1`, `alpha` for Tukey window

		* also `t0`, `t1` for truncating signals 

				t0=3e-3
				t1=13e-3

	* masker name for rough estim ur0

	* `gauss_sigma` (smoothing for signals, necessary for deconv)

	* roll for ur0

	* `t0`, `t1` for projection rough estim of R_0

	* latency values for narrow-band analysis (manual pick)

	* check that no errors for fitting latencies with power law (possibly, need to exclude outliers)



### Fitting procedure

`Fit data.ipynb`

* needs to import scripts for data pre-processing (`fit_data_common.py`)
* NB: `plot_figures_narrowband_analysis_deconv()` can help to find values for the latencies for the narrowband analysis method
* shift latencies
	* reference (0 ms): click for latest experiments, reference was CM for earliest experiments
* estim I/O func : init params are in `init_params/[expe_name]/[CF]_init_params.json` or hard coded in jupyter notebook
  * `utils/test weibull.ipynb` can help to find suitable init  params. or interact mode with plots can be used in main notebook (after optim, uncomment @interact_manual)
  * ex of init file:

```json
{
"I0": -11,
"scale": 30,
"k": 4
}
```

 * optim params: in `optim_params_[expe_name].json` or `optim_params.json`

#### Multiples CFs

* `run_distributed.ipynb` provides `papermill` commands (with `screen`)
* advise for the whole procedure:
	* first run with '_distributed' params deactivated 
		* in particular: first estim of I/O functions
	* second run with '_distributed' params activated  + `load_wbcdf=True`
		* you can try `I0_distributed` True (advised: `plus_lambda=True` the distrib param corresponds to 63% value of max wb cdf) or False
		* for E0 distrib, an important parameter is `n_dim`
		* if distrib is activated, `RBF.ipynb` has to be ran before the screen commands (RBF works with `distrib_params.json` which is generated by `run_distributed.ipynb`)
* `run_distributed.ipynb` export param files in results_folder specific to the run, but you'll need to copy the .ipynb files `fitdata[CF].ipynb` in the results folder if you want to keep track of optim plots.

### Analyzing results

* Some results are saved by `Fit data.ipynb`  (`write_results` on True), or plotted directly in the jupyter notebook
* Further analysis of data is done in `Fit data synthesis.ipynb`



Code for specific exper. data
------

Code for pre-processing, as well as optimization parameters (and sometimes initialization parameters) have to be adapted for each experimental dataset. As an example, we provide the specific code for the main figures (chinchilla Q395) that were presented in the paper (e.g., file `fit_data_1_22_common.py`). The jupyter notebook (`Fit data.ipynb`) loading this dataset should run by itself without modification.

