function superLearn(xTrain,labelsTrain,xTest,labelsTest, OPT)

models = learn(xTrain, labelsTrain, OPT);
n = 1;
newLabels = prediction(xTest, models, n);

u=unique(labelsTest);

for i=1:1
    G1vAll(:,i)=(labelsTest==u(i));
    e = abs(G1vAll(:,i)-newLabels{i});
    c(i) = length(e(e==0))/length(e)*100;
end

actual = [G1vAll(:,1)];
predicted = [newLabels{1}];
figure;
plotconfusion(predicted',actual');
title(['Total accuracy: ',num2str(mean(c)),'%']);

end