function plotHpNoiseCAPs(begin_ind, end_ind)
    %plot CAPs (baseline : no masker 'broadband_noise')
    % corresponding to high-pass noise maskers (narrowband analysis)
    % for pics between begin_ind and end_ind (optional parameters)
    %Data can be distributed on several pic files but filenames should
    %correspond to the same pattern (see code).
    %Extra atten (for maskers) assumed to be 0.
    
    %% Create mappings
    %data_folder='../../Data/Data-10-09/matFiles/'; %test
    data_folder=cd;

    if ~exist('begin_ind','var')
     % third parameter does not exist, so default it to something
      begin_ind = 0;
    end
    
    if ~exist('end_ind','var')
     % third parameter does not exist, so default it to something
      end_ind = Inf;
    end
 
    %values for test
    %begin_ind=175;   %27
    %end_ind=912;  %49
    
    validPic = @(n) (n>=begin_ind && n<=end_ind);

    files=dir(data_folder);
    exp0='p(?<picNumber>[0-9]{4})_*.mat';
    expBroadband='p(?<picNumber>[0-9]{4})_.*broadband_noise.mat';
    exp='p(?<picNumber>[0-9]{4})_fmasked_CAP_.*hp_(?<freq>.*?)Hz.*.mat';
    
    picDic=struct();
    picFiles=cell(1, length(files)); %list pic-> filename
    broadbandPic=[];
    
    freqfields={};
    
    for i=1:length(files)
        filename=files(i).name;
        m=regexp(filename, exp, 'names');
        if ~isempty(m) && validPic(str2num(m.picNumber))
            if ~ismember(m.freq, freqfields)
                freqfields{end+1}=m.freq;
            end
            freq_field=['fc_' m.freq];
            if ~isfield(picDic, freq_field)
                tmp=[str2num(m.picNumber)];
                picDic.(freq_field)= tmp;
            else

                picDic.(freq_field)=[picDic.(freq_field), str2num(m.picNumber)];               
            end

        else %more inclusive regexps
            m = regexp(filename, expBroadband, 'names');
            if ~isempty(m) && validPic(str2num(m.picNumber))
                broadbandPic=[broadbandPic str2num(m.picNumber)];
            else
                m=regexp(filename, exp0, 'names');
            end
        end

        %fill picFiles
        if ~isempty(m)
            picNumber=str2num(m.picNumber);
            picFiles(picNumber)={filename};
        end
    end

    %% Retrieve array for broadband

    firstPic=true;
    assert(~isempty(broadbandPic), 'no pic associated with broadband_noise found')
    for picNumber=broadbandPic
       %load pic
       filename=picFiles{picNumber};
       picStruct=load([data_folder '/' filename]);
       if firstPic
           %arr=picStruct.valAvg;
           arr=picStruct.data_struct.AD_Data.AD_Avg_V;
           firstPic=false;
       else
           %arr=arr+picStruct.valAvg;
           arr=arr+picStruct.data_struct.AD_Data.AD_Avg_V;
       end
    end
    arr=arr/length(broadbandPic);
    broadband_sig=arr;
    
    %% sort freqs
    
    freqs=zeros(1, length(freqfields));
    for i=1:length(freqfields)
        freq_field=freqfields{i};
        freq=str2num(freq_field);
        freqs(i)=freq;
    end
    [~, idx_sorted]=sort(-freqs);
    
    freqfields2=cell(1, length(freqfields));
    %% Plot curves
    for indfreq=1:length(freqfields)
        ifreq=idx_sorted(indfreq);
        freq_field=['fc_' freqfields{ifreq}];
        freqfields2{indfreq}=[freqfields{ifreq} ' Hz'];
        firstPic=true;
        for picNumber=picDic.(freq_field)
           %load pic
           filename=picFiles{picNumber};
           picStruct=load([data_folder '/' filename]);
           if firstPic
               %arr=picStruct.valAvg;
               arr=picStruct.data_struct.AD_Data.AD_Avg_V;
               %t=linspace(0,  picStruct.CAPlength_ms, length(arr));
               t=linspace(0,  picStruct.data_struct.Stimuli.CAP_intervals.CAPlength_ms, length(arr));

               firstPic=false;
           else
               %arr=arr+picStruct.valAvg;
               arr=arr+picStruct.data_struct.AD_Data.AD_Avg_V;
           end
        end
        arr=arr/length(picDic.(freq_field));

        if indfreq==1
            fig_raw=figure();
        else
            figure(fig_raw)
        end
        plot(t, arr);
        hold on;
        
        
        if indfreq==1
            fig2=figure();
           
        else
            figure(fig2)
        end
        plot(t, arr-broadband_sig);
        hold on;
    end
    figure(fig_raw)
    title('CAP HP noises (raw)');        
    xlabel('t (ms)')
    legend(freqfields2)
    figure(fig2)
    title('CAP HP noises (ref:broadband noise)')        
    xlabel('t (ms)')
    legend(freqfields2)
end