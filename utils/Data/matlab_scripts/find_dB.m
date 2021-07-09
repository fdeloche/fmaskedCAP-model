p=load('p0005_fmasked_CAP_teststim.mat');

fs=p.data_struct.Stimuli.RPsamprate_Hz;

XstartPlot_ms=p.data_struct.Stimuli.CAP_intervals.XstartPlot_ms;

XendPlot_ms=p.data_struct.Stimuli.CAP_intervals.XstartPlot_ms;

fprintf('Masker atten. : %0.2f \n', p.data_struct.Stimuli.masker_atten_dB)
fprintf('Probe atten. : %0.2f \n', p.data_struct.Stimuli.atten_dB)


All_V=p.data_struct.AD_Data.AD_All_V-mean(p.data_struct.AD_Data.AD_All_V, 'all');

% Plot
figure();
for i=1:5
   arr=All_V(i, :);
   plot( (1:length(arr))*1e3/fs, arr)
   hold on;
    
end
xlabel('t (ms)')

%noise RMS
%start/end points are entered manually (in ms):
t_0=70;
t_1=78; 

ind_0=round(t_0*fs/1000);
ind_1=round(t_1*fs/1000);

rms_n=mean(All_V(:, ind_0:ind_1).^2, 2); %avg on t
rms_n=mean(rms_n); %avg on reps
rms_n=sqrt(rms_n);

fprintf('RMS value noise : %.5f\n', rms_n);



%RMS masker

%start/end points are entered manually (in ms):
t_0=20;
t_1=45; 

ind_0=round(t_0*fs/1000);
ind_1=round(t_1*fs/1000);

rms_m=mean(All_V(:, ind_0:ind_1).^2, 2); %avg on t
rms_m=mean(rms_m); %avg on reps
rms_m=sqrt(rms_m);

fprintf('RMS value masker : %.5f\n', rms_m);

%Amplitude click

%interval where the click is searched (in ms):
t_2=62;
t_3=70;


ind_2=round(t_2*fs/1000);
ind_3=round(t_3*fs/1000);

peak_val= max( abs(All_V(:, ind_2:ind_3)) , [], 2);
fprintf('RMS value peak (probe) : %.5f\n', sqrt(mean(peak_val.^2) ));




%plot spectrum / multitaper

figure()
% 
% freq_res= fft(All_V(:, ind_0:ind_1), [], 2);

f=linspace(0, fs, ind_1-ind_0+1);

% %plot(f, abs(freq_res))
% plot(f, sqrt(mean(abs(freq_res).^2, 1)) )

nPairs=p.data_struct.Stimuli.RunStimuli_params.nPairs;
m_tapers=12;
sumrep=0;
for i=1:2*nPairs
    x=All_V(i, ind_0:ind_1);
    rep = pmtm(x,m_tapers);
    sumrep= sumrep + rep;
    
    f=linspace(0, fs/2, length(rep));
    plot(f, rep)
    hold on
end

plot(f, sumrep/nPairs)

