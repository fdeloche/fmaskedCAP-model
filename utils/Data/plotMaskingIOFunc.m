function plotMaskingIOFunc(begin_ind, end_ind)
    %plot CAP amplitude (baseline : no masker 'broadband_noise')
    % as a function of notch attenuation for different CFs
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
    %exp='p(?<picNumber>[0-9]{4})_fmasked_CAP_.*notch(?<freq>.*?)_(?<bw>.*?)_attn(?<attn>.*?)dB.mat';
    exp='p(?<picNumber>[0-9]{4})_fmasked_CAP_.*notch(?<freq>.*?)_(?<bw>.*?)_(?<attn>.*?)dB.mat';
    
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
            attn_field=['attn_' m.attn];
            if ~isfield(picDic, freq_field)
                tmp=struct();
                tmp.(attn_field)=[str2num(m.picNumber)];
                picDic.(freq_field)= tmp;
            else
                if ~isfield(picDic.(freq_field), attn_field)
                    picDic.(freq_field).(attn_field)=[str2num(m.picNumber)];               
                else
                    picDic.(freq_field).(attn_field) = [picDic.(freq_field).(attn_field) str2num(m.picNumber)];
                end
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

    %% Plot curves
    for ifreq=1:length(freqfields)
        freq_field=['fc_' freqfields{ifreq}];

        figure()
        attn_fields=fieldnames(picDic.(freq_field));
        labels=cell(1, length(attn_fields));
        %sort attns
        attns=zeros(1, length(attn_fields));
        for i=1:length(attn_fields)
            attn_field=attn_fields{i};
            m=regexp(attn_field, 'attn_(?<attn>.*)', 'names');
            attn=str2num(m.attn);
            attns(i)=attn;
        end
        [~, idx_sorted]=sort(-attns);

        max_arr=zeros(1, length(attn_fields));

        min_arr=zeros(1, length(attn_fields));
        for k=1:length(idx_sorted)
            i=idx_sorted(k);
            attn_field=attn_fields{i};
            m=regexp(attn_field, 'attn_(?<attn>.*)', 'names');
            attn_st=m.attn;
            attn=str2num(attn_st);
            labels{k}=['attn ' attn_st ' dB'];
            firstPic=true;
            for picNumber=picDic.(freq_field).(attn_field)
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
            arr=arr/length(picDic.(freq_field).(attn_field));
            max_arr(i)=max(arr-broadband_sig);
            min_arr(i)=min(arr-broadband_sig);

            plot(t, arr-broadband_sig);
            hold on;
        end
        title(['CAP notch '  freqfields{ifreq} ' Hz']);
        legend(labels)
        xlabel('t (ms)')

        CAP_arr=max_arr-min_arr;
        
        if ifreq==1
            fig_synth=figure();
        else
            figure(fig_synth)
        end
        plot(attns(idx_sorted), CAP_arr(idx_sorted));
        hold on;
        xlabel('Notch attenuation (dB)')
        ylabel('CAP amplitude (ref: broadband 20dB attn)')
    end
    legend(freqfields)
end