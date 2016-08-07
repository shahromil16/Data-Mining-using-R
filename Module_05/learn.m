function models = learn(X,labels,opt)

u=unique(labels);
n=length(u);

if opt==1
    for k=1:n
        G1vAll=(labels==u(k));
        models{k} = fitcsvm(X,G1vAll);
    end
elseif opt==2
    for k=1:n
        G1vAll=(labels==u(k));
        models{k} = fitcnb(X,G1vAll);
    end
elseif opt==3
    for k=1:n
        G1vAll=(labels==u(k));
        models{k} = fitcknn(X,G1vAll);
    end
end