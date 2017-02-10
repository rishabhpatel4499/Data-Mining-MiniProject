clear;
X_train = importdata('Multi class VidTIMIT Data/X_train.mat');
y_train = importdata('Multi class VidTIMIT Data/y_train.mat');
X_test = importdata('Multi class VidTIMIT Data/X_test.mat');
y_test = importdata('Multi class VidTIMIT Data/y_test.mat');
y_train = transpose(y_train);
%y_test = transpose(y_test);
 temp = zeros(3500,1);
 res = zeros(1000,25);
for i=1:25
    for j = 1:numel(y_train)
        if(y_train(j) == i)
            temp(j) = 1;
        else
            temp(j) = -1;
        end
    end
    mdl = fitcsvm(X_train, temp, 'KernelFunction', 'polynomial', 'Polynomialorder' , 2);
    label = predict(mdl, X_test);
    res(1:1000,i) = label;
end  
label = zeros(1,1000);
for i=1:1000
    for j=1:25
        if(res(i,j)==1)
            label(i)=j;
        end
    end
end
accuracy = sum(y_test == label) / numel(y_test);
accuracyPercentage = 100*accuracy;
fprintf('\n\nOverall Accuracy : %f%%\n',accuracyPercentage);