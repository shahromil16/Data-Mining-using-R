function newLabels = prediction(X, models, n)

d = size(X,1);

for k=1:n
    [newLabels{k},scores{k}] = predict(models{k}, X);
end