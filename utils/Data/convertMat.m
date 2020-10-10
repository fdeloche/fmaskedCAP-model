clear all;
for nPic=736:924
    struct = loadPic(nPic);
    date=struct.General.date;
    time=struct.General.time;
    atten_dB=struct.Stimuli.atten_dB;
    masker_atten_dB=struct.Stimuli.masker_atten_dB;
    fixedPhase=struct.Stimuli.fixedPhase;
    fs=struct.Stimuli.RPsamprate_Hz;
    nPairs=struct.Stimuli.RunStimuli_params.nPairs;
    invFilterOnWavefiles=struct.Stimuli.RunStimuli_params.invFilterOnWavefiles;
    lpcInvFilterOnClick=struct.Stimuli.RunStimuli_params.lpcInvFilterOnClick;
    decimateFact=struct.Stimuli.RunStimuli_params.decimateFact;
    duration_masker_ms=struct.Stimuli.CAP_intervals.duration_ms;
    rftime_ms=struct.Stimuli.CAP_intervals.rftime_ms;
    period_ms=struct.Stimuli.CAP_intervals.period_ms;
    XstartPlot_ms=struct.Stimuli.CAP_intervals.XstartPlot_ms;
    XendPlot_ms=struct.Stimuli.CAP_intervals.XendPlot_ms;
    CAPlength_ms=struct.Stimuli.CAP_intervals.CAPlength_ms;
    clickDelay=struct.Stimuli.CAP_intervals.clickDelay;
    masker_name=struct.Stimuli.masker.name;
    n_bands=struct.Stimuli.masker.n_bands;
    bands=struct.Stimuli.masker.bands;
    wavefilename=struct.Stimuli.masker.wavefilename_used;
    assert(struct.Line.atten_dB==atten_dB)
    masker_amp=struct.Line.maskerAmp;
    
    valAvg=struct.AD_Data.AD_Avg_V;
    gain=struct.AD_Data.Gain;
    valAll=struct.AD_Data.AD_All_V;
    filename=getFileName(nPic);
    filename=strrep(filename, '.m', '.mat');
    %filename=strrep(filename, '.m', '.npy');
    %mat2np(val,filename, 'float64');
    clear struct;
    save(['matFiles/' filename])

    %save(['matFiles/' filename], 'Avg')
end