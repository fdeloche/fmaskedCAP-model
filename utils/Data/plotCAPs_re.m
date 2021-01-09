function figs_labels = plotCAPs_re(re, figs, labels,begin_ind, end_ind)
    %plot CAPs (baseline : no masker 'broadband_noise')
    % corresponding to regular expression re   ex: '_.*broadband_noise'
    % the regular expression considered will be
    % 'p(?<picNumber>[0-9]{4})_.*broadband_noise.mat'
    % for pics between begin_ind and end_ind (optional parameters)
    %Extra atten (for maskers) assumed to be the same.
    %figs: optional argument, {fig_raw, fig} figures on which CAPs are
    %labels: list of labels for existing plots
    %plotted
    
    %% Create mappings
    %data_folder='../../Data/Data-01-07-2021-test-extra-atten'; %test
    data_folder=cd;

    if ~exist('begin_ind','var')
     % third parameter does not exist, so default it to something
      begin_ind = 0;
    end
    
    if ~exist('end_ind','var')
     % third parameter does not exist, so default it to something
      end_ind = Inf;
    end
    
    if ~exist('figs','var')
     % third parameter does not exist, so default it to something
      fig_raw= figure();
      fig2=figure();
    else
        fig_raw=figs{1};
        fig2=figs{2};
    end
    
    
    if ~exist('labels','var')
        labels={};
    end

    validPic = @(n) (n>=begin_ind && n<=end_ind);

    files=dir(data_folder);
    exp0='p(?<picNumber>[0-9]{4})_(?<name>.*).mat';
    expBroadband='p(?<picNumber>[0-9]{4})_.*broadband_noise.mat';
    exp=['p(?<picNumber>[0-9]{4})' re '.mat'];
    
    picFiles=cell(1, length(files)); %list pic-> filename
    broadbandPic=[];
    
    picDic={};
    names={}
    for i=1:length(files)
        filename=files(i).name;
        m=regexp(filename, exp, 'names');
        if ~isempty(m) && validPic(str2num(m.picNumber))
               m=regexp(filename, exp0, 'names');
               name=m.name;
                if ~ismember(name, names)
                    names{end+1}=m.name;
                end
                if ~isfield(picDic, name)
                    tmp=[str2num(m.picNumber)];
                    picDic.(name)= tmp;
                else

                    picDic.(name)=[picDic.(name), str2num(m.picNumber)];               
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
    
    firstPic=true;
    for i=1:length(names)
        name=names{i};
        
        labels{end+1}=name;
        listPicNum=picDic.(name);
        for picNumber=listPicNum
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
        arr=arr/length(listPicNum);

       figure(fig_raw)


        plot(t, arr);
        hold on;
        
        
        figure(fig2)
        plot(t, arr-broadband_sig);
        hold on;
    end


    figure(fig_raw)
    title('CAPs (raw)');        
    xlabel('t (ms)')
    leg=legend(labels);
    set(leg,'Interpreter', 'none')
    figure(fig2)
    title('CAP HP noises (ref:broadband noise)')        
    xlabel('t (ms)')
    leg=legend(labels);
    set(leg,'Interpreter', 'none')
    
    figs2={fig_raw, fig2};
    figs_labels={figs2, labels};
end