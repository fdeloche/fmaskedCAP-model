data_folder='../../Data/Data-10-09/matFiles';
files=dir(data_folder);
exp='p(?<picNumber>[0-9]{4})_fmasked_CAP_.*notch(?<freq>.*?)_(?<bw>.*?)_attn(?<attn>.*?)dB.mat';
picDic=struct();
for i=1:length(files)
    filename=files(i).name;
    m=regexp(filename, exp, 'names');
    if ~isempty(m)
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
    end
end


%% Ex
freq=2500;

freq_field=['fc_' num2str(freq)];
attn_fields=fieldnames(picDic.(freq_field));
for i=1:length(attn_fields)
    attn_field=attn_fields{i};
    m=regexp(attn_field, 'attn_(?<attn>.*)', 'names');
    attn=m.attn;
    firstPic=true;
    for picNumber=picDic.(freq_field).(attn_field)
       if firstPic
           firstPic=false;
       end
    end
end
