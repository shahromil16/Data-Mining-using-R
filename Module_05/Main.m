clear all
clc

presdir = pwd;

%%
cd(strcat(presdir,'\Parkinson_Multiple_Sound_Recording_Data'));
dataTrain = dlmread('train_data.txt',',');
dataTest = dlmread('test_data.txt',',');

labelsTrain = dataTrain(:,end);
labelsTest = dataTest(:,end);

cd(presdir);

tic;
superLearn(dataTrain(:,2:27),labelsTrain,dataTest(:,2:27),labelsTest, 1)
time1 = toc;
tic;
superLearn(dataTrain(:,2:27),labelsTrain,dataTest(:,2:27),labelsTest, 2)
time2 = toc;
tic;
superLearn(dataTrain(:,2:27),labelsTrain,dataTest(:,2:27),labelsTest, 3)
time3 = toc;
tic;
unsuperLearn(dataTrain(:,2:27),labelsTrain, 1);
time4 = toc;

%%
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

superLearn(ankles{1},labels{1},ankles{2},labels{2}, 3)
unsuperLearn(ankles{1},labels{1}, 1);

ix = random('unif',0,1,size(ankles{1}))<0.30; 
ankles{1}(ix) = NaN;
tic;
superLearn(ankles{1},labels{1},ankles{2},labels{2}, 3);
time5 = toc;
tic;
unsuperLearn(ankles{1},labels{1}, 1);
time6 = toc;
[coeff1,score1,latent,tsquared,explained,mu1] = pca(ankles{1},'algorithm','als');
t = score1*coeff1' + repmat(mu1,size(ankles{1},1),1);
tic;
superLearn(t,labels{1},ankles{2},labels{2}, 3);
time7 = toc;
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