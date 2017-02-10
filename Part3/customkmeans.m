function [Cluster, sse] = customkmeans(X, K)
N = size(X,1);
iterations = 0;
sse_prev = 0;
sse = 0;
CentroidsMatrix = X(randsample(N, K), :);

while 1
	iterations = iterations + 1;
	d = pdist2(CentroidsMatrix, X, 'euclidean');
	[NearClusterDist, Cluster] = min(d, [], 1);
	sse = 0;
	for i=1:K
		indx = Cluster == i;
		% dis = pdist2(CentroidsMatrix(i,:), X(indx,:),'euclidean')
		dis = NearClusterDist(indx);
		sse = sse + sumsqr(dis);
	end
	fprintf('sse of loop %d is = %f\n', iterations, sse);

    if iterations >= 100 ||  abs(sse - sse_prev) <= 0.001
    	break
    end
	sse_prev = sse;

	for i=1:K
	    indx = Cluster == i;
	    CentroidsMatrix(i,:) = mean(X(indx, :),1);
	end
end
