

## Description 

Code for *Estimation of cochlear frequency selectivity based on a convolution model of forward-masked compound action potentials*. François Deloche, Satyabrata Parida, Andrew Sivaprakasam,  Michael G. Heinz. 2022

Zenodo badge: [![DOI](https://zenodo.org/badge/476437527.svg)](https://zenodo.org/badge/latestdoi/476437527)

## Lab notes 

Some of these notes are specific to the data pipeline in our lab, but explains nevertheless the entire procedure from handling raw data to analysis of results.

### Steps fitting new expe

#### Pre- fitting procedure

* expe Data (.mat files) in `Data/[expe-name]/` . picture files: format similar to CAP data in the Heinz lab (+ infos on masker).

  * you'll need: `data_folder `, `begin_ind`, `end_ind` (start/end pic numbers; you can possibly use `pic_numbers_ignore`)

* In main folder: create `fit_data_list_maskers.py` from example with list of maskers. For each CF:

	* `ntch`: notched noise maskers with diff atten for the notch (I/O function)
	* `vbw` various bandwidths (notch widths, estim of 10dB bandwidth)
	* `vfreq` various frequencies (for the notch) around CF (more maskers for the estim of frequency weights)

* In main folder: create `fit_data_common.py` from example. Contains code/info for setting reference intensity, pre-processing CAP responses, rough estim of R_0 (deconv),  narrow-band analysis and estimation of latencies. Parameters to change (possibly):

	* `data_folder`

	* `I0` based on NEL output (says something like 'max output should be 105 dB for RMS=1') + masker atten

	* `begin_ind`, `end_ind` , `pic_numbers_ignore`

	* `t0`, `t1`, `alpha` for Tukey window

		* also `t0`, `t1` for truncating signals 

				t0=3e-3
				t1=13e-3

	* masker name for rough estim ur0

	* `gauss_sigma` (smoothing for signals, necessary for deconv)

	* roll for ur0

	* `t0`, `t1` for projection rough estim of R_0

	* enter latency values using narrow-band analysis 

	* check that no errors for fitting latencies with power law (possibly, need to exclude outliers)



### Fitting procedure

Using `Fit data.ipynb`

* import files created in previous section
* NB: you can run `plot_figures_narrowband_analysis_deconv()` to help you find values for latencies
* shift latencies
	* reference click for later experiments, reference CM for earlier experiments
* make sure estim ur ok (with proj)
* estim I/O func : you can change init params in `init_params/[expe_name]/[CF]_init_params.json` or hard coded in jupyter notebook
	* you can use `utils/test weibull.ipynb` to help you find good params. or you can use interact mode with  plots in main notebook (after optim, uncomment @interact_manual)
	* ex file:

```json
{
"I0": -11,
"scale": 30,
"k": 4
}
```

 * optim params: in `optim_params_[expe_name].json` or `optim_params.json`

#### Multiples CFs

* You can use `run_distributed.ipynb` which provides `papermill` commands (with `screen`)
* advise:
	* first run with '_distributed' params deactivated 
		* in particular: first estim of I/O functions
	* second run with '_distributed' params activated  + `load_wbcdf=True`
		* you can try `I0_distributed` True (advised: `plus_lambda=True` the distrib param corresponds to 63% value of max wb cdf) or False
		* for E0 distrib, an important parameter is `n_dim`
		* if distrib is activated, you'll have to run `RBF.ipynb`before the screen commands (RBF works with `distrib_params.json` which is generated by `run_distributed.ipynb`)
* `run_distributed.ipynb` copy param files in results_folder specific to the run, but you need to copy the .ipynb files `fitdata[CF].ipynb` in the results folder if you want to keep track of optim plots.

### Analyzing results

* Some results are saved by `Fit data.ipynb`  (`write_results` on True), or plotted directly in the jupyter notebook
* Further analysis of data is done in `Fit data synthesis.ipynb`

