function newlabels = unsuperLearn(dataTrain,labelsTrain, OPT)

if OPT == 1
    newlabels = kmeans(dataTrain,2);
    newlabels(newlabels==1)=1;
    newlabels(newlabels==2)=0;
    
    u=unique(labelsTrain);
    
    G1vAll(:)=(labelsTrain==u(1));
    e = abs(G1vAll(:)-newlabels);
    c = length(e(e==0))/length(e)*100;
    
    actual = G1vAll(:);
    predicted = newlabels;
    figure;
    plotconfusion(predicted',actual');
    title(['Total accuracy: ',num2str(mean(c)),'%']);
    
elseif OPT == 2
    [~, newlabels, ~, ~] = ExMax(dataTrain, 2, 1e-4, 1000);
    
end