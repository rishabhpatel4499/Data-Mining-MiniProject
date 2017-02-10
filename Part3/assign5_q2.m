clear;
X_train = importdata('Handwritten Digits/X_train.mat');
y_train = importdata('Handwritten Digits/y_train.mat');
X_test = importdata('Handwritten Digits/X_test.mat');
y_test = importdata('Handwritten Digits/y_test.mat');


knn_model = fitcknn(X_train, y_train,'NumNeighbors',7);
label = predict(knn_model,X_test);
classList=unique(y_test);

label_t = transpose(label);ytest1=transpose(y_test);
accuracy = sum(ytest1 == label_t) / numel(ytest1);
accuracyPercentage = 100*accuracy;
fprintf('\n\nOverall Accuracy (knn) : %f%%\n',accuracyPercentage);

y_train1 = transpose(y_train);
%y_test = transpose(y_test);
 temp = zeros(500,1);
 res = zeros(3251,10);
for i=1:10
    for j = 1:numel(y_train1)
        if(y_train1(j) == i)
            temp(j) = 1;
        else
            temp(j) = -1;
        end
    end
    mdl = fitcsvm(X_train, temp, 'KernelFunction', 'polynomial', 'Polynomialorder' , 2);
    label = predict(mdl, X_test);
    res(1:3251,i) = label;
end  
label = zeros(1,3251);
for i=1:3251
    for j=1:10
        if(res(i,j)==1)
            label(i)=j;
        end
    end
end
accuracy = sum(transpose(y_test) == label) / numel(transpose(y_test));
accuracyPercentage = 100*accuracy;
fprintf('\n\nOverall Accuracy (svn) : %f%%\n',accuracyPercentage);



target  = full(ind2vec(transpose(y_train))) ;
net = patternnet(25);
net = train(net,transpose(X_train),target);
y = net(X_test');
classes = vec2ind(y);
dataClassifiedAccurately = (classes==transpose(y_test));
percentageAccuracy=sum(dataClassifiedAccurately)/length(dataClassifiedAccurately)*100;
fprintf('\n\nOverall Accuracy (ann) : %f%%\n',percentageAccuracy);

ensemble = [label_t;label;classes];
result = zeros(3251,1);
for i=1:3251
    result(i) = mode(ensemble(:,i));
end 
dataClassifiedAccurately = (result==y_test);
percentageAccuracy=sum(dataClassifiedAccurately)/length(dataClassifiedAccurately)*100;
fprintf('\n\nOverall Accuracy (ensemble) : %f%%\n',percentageAccuracy);

