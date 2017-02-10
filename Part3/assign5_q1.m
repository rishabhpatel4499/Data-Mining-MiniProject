clear;
X = importdata('x.txt');
[Cluster1, sse1] = customkmeans(X, 3);
[Cluster2, sse2] = customkmeans(X, 5);
[Cluster3, sse3] = customkmeans(X, 7);
fprintf('\n\nk=3 sse= : %f%%\n',sse1);
fprintf('\n\nk=5 sse= : %f%%\n',sse2);
fprintf('\n\nk=7 sse= : %f%%\n',sse3);