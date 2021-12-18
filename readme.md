### Steps fitting new expe

#### Pre- fitting procedure

* expe Data (.mat files) in `Data/[expe-name]/` . picture files: format similar to CAP data in the Heinz lab (+ infos on masker).

* Explore data: see examples in `utils/Data`

	* you'll need: `data_folder `, `begin_ind`, `end_ind` (start/end pic numbers; you can possibly use `pic_numbers_ignore`)

* In main folder: create `fit_data_[expe_date]_list_maskers.py` from example with list of maskers. For each CF:

	* `ntch`: notched noise maskers with diff atten for the notch (I/O function)
	* `vbw` various bandwidths (notch widths, estim of 10dB bandwidth)
	* `vfreq` various frequencies (for the notch) around CF (more maskers for the estim of frequency weights)

* In main folder: create `fit_data_[expe_date]_common.py` from example. Contains code/info for setting reference intensity, pre-processing CAP responses, rough estim of R_0 (deconv),  narrow-band analysis and estimation of latencies. Parameters to change (possibly):

	* `data_folder`

	* `I0` based on NEL output (says something like 'max output should be 105 dB for RMS=1') + masker atten

	* `begin_ind`, `end_ind` , `pic_numbers_ignore`

	* `t0`, `t1`, `alpha` for Tukey window

		* also `t0`, `t1` for truncating signals but normally no need to change

				t0=3e-3
				t1=13e-3

	* masker name for rough estim ur0

	* `gauss_sigma` (smoothing for signals, necessary for deconv)

	* roll for ur0

	* `t0`, `t1` for projection rough estim of R_0

	* report values for narrow-band analysis (using deconv?)

	* check that no errors for fitting latencies with power law (possibly, you need to exclude outliers



### Fitting procedure

Using `Fit data.ipynb`

* import files created in previous section
* NB: you can run `plot_figures_narrowband_analysis_deconv()` to help you report values for latencies
* shift latencies
	* reference click for later experiments, reference CM for earlier experiments
* 