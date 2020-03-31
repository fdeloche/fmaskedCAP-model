---
title: Draft project on CAP modeling
author: François Deloche
date: March 19, 2020
---

### Elements from previous work


Note: Good review article on CAP  
*Ups and downs in 75 years of electrocochleography* Eggermont [@Eggermont2017].

#### The narrow-band contributions to the compound action potential

Masking of a click or tone-burst with a high-passed white noise (NB: not strictly speaking forward masking as the noise was kept during the probe, as far as I understand). The subtraction of the CAP obtained with masking with the previous recording ($f_{cut}$ cut-off frequency decreases at each step) gives the 'narrow-band' contribution of the CAP.


It has been used especially by Eggermont, although not the 1st one to use it (Teas et al., 1962)

* *Analysis of compound action potential responses to tone bursts in the human and guinea pig cochlea* Eggermont 1976 [@Eggermont1976]
  * TT recordings
* *Narrow-band analysis of compound action potentials for several stimulus conditions in the guinea pig* Eggermont 1981

![NAP of click in humans](./figures/NAP_click.png)





![NAP of tone burst in humans. Eggermont 1976](./figures/NAP_tone.png)



Note: He found quite a different pattern for NAP in response to tone bursts in guinea pigs (second negative peak):



![NAP of tone burst in guinea pigs. Eggermont 1976](./figures/NAP_tone_guinea.png)

 Also we seem to see broader tuning of auditory filters.

* more frequencies contribute to the CAP
*  also the 'narrow-band analysis' is less frequency selective. In fact, the 'NAPs' do not show exactly the contribution of each frequency band because of the spread of the masker along frequencies (this problem is alleviated in the model I propose because it seeks to estimate this spread) 





The NAP method was used to estimate the latencies of each frequency contribution:

![Latencies estimated with the NAP method](./figures/NAP_tone_latencies.png)



(exponential dependency, more visible on this figure for click : )



![Latencies estimated with the NAP method](./figures/NAP_click_latencies.png)



#### Unit responses, convolution models



* Individual ANF contribution

From Eggermont 2017 (review article):

> Further experimental evidence for the applicability of the NAP technique in pathological cochleas came from recordings in normal and noise-exposed guinea pigs (Versnel et al., 1992), which looked at the validity of using the same unit response along the CF range and in normal vs. hearing loss ears. They used a technique pioneered by Kiang et al. (1976) involving spike-triggered averaging of round window ‘‘noise’’. In that way one can estimate the unit response for units with CFs corresponding to locations along the cochlear partition. Their findings in normal cochleas confirmed the earlier data from Prijs (1986), namely that the unit response was diphasic and had a fairly constant amplitude of about 0.1 µV. In noise-exposed cochleas, waveform, latency and amplitude of the negative component of the unit response remained unchanged.

![Unit response estimated with spike-triggered average, Prijs 1986 [@Prijs1986] ](./figures/unit_response.png)



* Synchronous fibers contribution

Related to the PST histogram or time distribution of first spike (if 2nd peak negligible)

![Contributions of several ANFs at the onset of a tone. [@Ozdamar1978]](./figures/PST_tone.png)



* (double) convolution models
  * *Synthetic whole-nerve action potentials for the cat* E. de Boer 1975 (not completely read yet, but artifical PST histograms are computed with a filter + envelope + rectifier model, so it seems to go far in the modeling!)
  * 