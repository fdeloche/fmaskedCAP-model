function x = loadPic(picNum)     % Load picture
picSearchString = sprintf('a%04d*.m', picNum);

% %HG ADDED 2/20/20
% if contains(picSearchString,'raw')
%     return; %totally exit function
% end

picMFile = dir(picSearchString);
if (~isempty(picMFile))
    eval( strcat('x = ',picMFile.name(1:length(picMFile.name)-2),';') );

else 
    picSearchString = sprintf('p%04d*.m', picNum);
    picMFile = dir(picSearchString);
    if (~isempty(picMFile))
         eval( strcat('x = ',picMFile.name(1:length(picMFile.name)-2),';') );
    else
        error = sprintf('Picture file p%04d*.m not found.', picNum);
        x = [];
        return;
    end
end