clear;
X_train = importdata('Multi Label Scene Data/X_train.mat');
y_train = importdata('Multi Label Scene Data/y_train.mat');
X_test = importdata('Multi Label Scene Data/X_test.mat');
y_test = importdata('Multi Label Scene Data/y_test.mat');

labels = [];
for i=1:6
    mdl = fitcsvm(X_train, y_train(:, i:i), 'KernelFunction', 'polynomial', 'Polynomialorder' , 2);
    labels = [labels predict(mdl, X_test)];
end


intersec = bsxfun(@and, labels, y_test);
uni = bsxfun(@or, labels, y_test);

sum_intersec = sum(intersec, 2);
sum_uni = sum(uni, 2);
sim = bsxfun(@rdivide, sum_intersec, sum_uni);
fprintf('\nAccuracy1:%f',mean(sim)*100);


labels2 = [];
for i=1:6
    mdl = fitcsvm(X_train, y_train(:, i:i), 'KernelFunction', 'Gaussian', 'KernelScale', 'auto' );
    labels2 = [labels2 predict(mdl, X_test)];
end


intersec = bsxfun(@and, labels2, y_test);
uni = bsxfun(@or, labels2, y_test);

sum_intersec = sum(intersec, 2);
sum_uni = sum(uni, 2);
sim2 = bsxfun(@rdivide, sum_intersec, sum_uni);
mean(sim2);
fprintf('\nAccuracy2:%f',mean(sim2)*100);