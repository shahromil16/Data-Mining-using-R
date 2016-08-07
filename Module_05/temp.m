clear all
clc

presdir = pwd;

%%
cd(strcat(presdir,'\Parkinson_Multiple_Sound_Recording_Data'));
dataTrain = dlmread('train_data.txt',',');
dataTest = dlmread('test_data.txt',',');

labelsTrain = dataTrain(:,end); labelsTrain(end-51:end) = ones(52,1);
labelsTest = dataTest(:,end);

cd(presdir);

%SVM
tic;
superLearn(dataTrain(:,2:27),labelsTrain,dataTest(:,2:27),labelsTest, 1)
time1 = toc;
%Naive
tic;
superLearn(dataTrain(:,2:27),labelsTrain,dataTest(:,2:27),labelsTest, 2)
time2 = toc;
%k-NN
tic;
superLearn(dataTrain(:,2:27),labelsTrain,dataTest(:,2:27),labelsTest, 3)
time3 = toc;
%k-Means
tic;
unsuperLearn(dataTrain(:,2:27),labelsTrain, 1);
time4 = toc;

%%
clear all;
clc;

presdir = pwd;
cd(strcat(presdir,'\dataset_fog_release\dataset_fog_release\dataset'));
txtList = dir('*.txt');
txtNos = length(txtList);

for i=1:17
    i
    temp = load(txtList(i).name);
    data{i} = temp(temp(:,end)==1 | temp(:,end)==2,:);
    labels{i} = data{i}(:,end);
    ankles{i} = data{i}(:,2:4);
end

DD = ankles{1};

cd(presdir);
ankles{1} = DD;
ix = random('unif',0,1,size(ankles{1}))<0.01; 
ankles{1}(ix) = NaN;

%k-NN
superLearn(ankles{1},labels{1},ankles{2},labels{2}, 3);
%k-Means
unsuperLearn(ankles{1},labels{1}, 1);

tic;
superLearn(ankles{1},labels{1},ankles{2},labels{2}, 3);
time5 = toc;
tic;
unsuperLearn(ankles{1},labels{1}, 1);
time6 = toc;
[coeff1,score1,latent,tsquared,explained,mu1] = pca(ankles{1},'algorithm','als');
t = score1*coeff1' + repmat(mu1,size(ankles{1},1),1);

%PCA + Naive
tic;
superLearn(t,labels{1},ankles{2},labels{2}, 3);
time7 = toc;
%PCA + k-Means
tic;
unsuperLearn(t,labels{1}, 1);
time8 = toc;

figure;
hold on;
for i=1:length(labels{1})
    i
    if labels{1}(i)==1
        plot(i,DD(i,1),'.b');
        hold on;
    else
        plot(i,DD(i,1),'or');
        hold on;
    end
end