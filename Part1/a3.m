clear;
X_train = importdata('VidTIMIT/X_train.mat');
y_train = importdata('VidTIMIT/y_train.mat');
X_test = importdata('VidTIMIT/X_test.mat');
y_test = importdata('VidTIMIT/y_test.mat');
target  = full(ind2vec(y_train)) ;
net = patternnet(25);
net = train(net,transpose(X_train),target);
y = net(X_test');
classes = vec2ind(y);
dataClassifiedAccurately = (classes==y_test);
percentageAccuracy=sum(dataClassifiedAccurately)/length(dataClassifiedAccurately)*100;
fprintf('\n\nOverall Accuracy : %f%%\n',percentageAccuracy);
