function picToWav(numPic)
    filename=getFileName(numPic);
    pic=load(filename);
    arr=pic.data_struct.AD_Data.AD_Avg_V;
    fs=pic.data_struct.Stimuli.RPsamprate_Hz;
    filename_cell_arr= strsplit(filename, '.');
    name=filename_cell_arr{1};
    if ~isfolder('./wavefiles/')
       mkdir('./wavefiles')
    end
    arr=arr-mean(arr);
    new_filename=['./wavefiles/' name '.wav'];
    amp_factor=100;
    audiowrite(new_filename,amp_factor*arr,fs);
end