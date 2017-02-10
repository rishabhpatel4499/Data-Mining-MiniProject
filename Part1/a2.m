clear;
%current_dir = pwd;
%files = dir([current_dir '\VidTIMIT']);
X_train = importdata('VidTIMIT/X_train.mat');
y_train = importdata('VidTIMIT/y_train.mat');
X_test = importdata('VidTIMIT/X_test.mat');
y_test = importdata('VidTIMIT/y_test.mat');

knn_model = fitcknn(X_train, y_train,'NumNeighbors',7);
label = predict(knn_model,X_test);
classList=unique(y_test);

label_t = transpose(label);
dataClassifiedAccurately = (label_t == y_test);
percentageAccuracy=sum(dataClassifiedAccurately)/length(dataClassifiedAccurately)*100;
fprintf('\n\nOverall Accuracy : %f%%\n',percentageAccuracy);




