function [IDX] = nearestneighbour(X, C)
    %NEARESTNEIGHBOUR nearest neighbourhood classification
    %   A = NEARESTNEIGHBOUR(X,C) finds the nearest row from C for each
    %   row of X. IDX is a column of the row indices for X.
    n = size(X,1);
    m = size(C,1);

    XC = X*C';          % inner products of data and centroids (n x m)
    CC = sum(C.*C,2);   % squared magnitude of centroids (m x 1)
    XX = sum(X.*X,2);   % squared magnitude of data (n x 1)
    D = repmat(CC',n,1) + repmat(XX,1,m) - 2*XC;   % squared distances
    [DMIN, IDX] = min(D,[],2); % minimum distances and indices

end



