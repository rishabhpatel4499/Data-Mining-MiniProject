clear;
X_train = importdata('Human Activity Recognition/X_train.txt');
y_train = importdata('Human Activity Recognition/y_train.txt');
X_test = importdata('Human Activity Recognition/X_test.txt');
y_test = importdata('Human Activity Recognition/y_test.txt');


knn_model = fitcknn(X_train, y_train,'NumNeighbors',7);
label = predict(knn_model,X_test);
classList=unique(y_test);

label_t = transpose(label);ytest1=transpose(y_test);
accuracy = sum(ytest1 == label_t) / numel(ytest1);
accuracyPercentage = 100*accuracy;
fprintf('\n\nOverall Accuracy : %f%%\n',accuracyPercentage);