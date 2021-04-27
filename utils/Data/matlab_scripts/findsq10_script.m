lpic=findPics('tc')
[thrs, bfs, q10s] = plotTCs(1, lpic, 0)   %note: had to put figure(XXX) in if statement
figure(); plot(bfs, q10s, '+'); 
mat=[lpic; bfs; q10s]
writematrix(mat', 'q10s.csv')


lpic2=findPics('SR');
quality=cell(2, length(lpic2));
for i = 1:length(lpic2)
    picNum=lpic2(i);
    pic=loadpic(picNum);
    quality{1, i}=picNum;
    quality{2, i}=pic.General.trigger;
end
