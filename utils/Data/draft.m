


%% Create mappings
data_folder='../../Data/Data-10-09/matFiles/';

begin_ind=175;   %27
end_ind=912;  %49
validPic = @(n) (n>=begin_ind && n<=end_ind);

files=dir(data_folder);
exp0='p(?<picNumber>[0-9]{4})_*.mat';
expBroadband='p(?<picNumber>[0-9]{4})_.*broadband_noise.mat';
exp='p(?<picNumber>[0-9]{4})_fmasked_CAP_.*notch(?<freq>.*?)_(?<bw>.*?)_attn(?<attn>.*?)dB.mat';
picDic=struct();
picFiles=cell(1, length(files)); %list pic-> filename
broadbandPic=[];
for i=1:length(files)
    filename=files(i).name;
    m=regexp(filename, exp, 'names');
    if ~isempty(m) && validPic(str2num(m.picNumber))
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
for picNumber=broadbandPic
   %load pic
   filename=picFiles{picNumber};
   picStruct=load([data_folder '/' filename]);
   if firstPic
       arr=picStruct.valAvg;
       firstPic=false;
   else
       arr=arr+picStruct.valAvg;
   end
end
arr=arr/length(broadbandPic);
broadband_sig=arr;

%% Ex
%freq=2500;
%freq_field=['fc_' num2str(freq)];

freq_field=['fc_5k'];

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
           arr=picStruct.valAvg;
           t=linspace(0,  picStruct.CAPlength_ms, length(arr));
           firstPic=false;
       else
           arr=arr+picStruct.valAvg;
       end
    end
    arr=arr/length(picDic.(freq_field).(attn_field));
    max_arr(i)=max(abs(arr-broadband_sig));
    plot(t*1e3, arr-broadband_sig);
    hold on;
end
title('CAP (ref: broadband noise 20 dB attn)');
legend(labels)
xlabel('t (ms)')


figure();
plot(attns(idx_sorted), max_arr(idx_sorted));

xlabel('Notch attenuation (dB)')
ylabel('CAP amplitude (ref: broadband 20dB attn)')

