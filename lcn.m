function [B] = lcn(A)

    %LCN Local Contrast Normalisation
    %   Remove variation in mean intensity over grey-scale image

    H2 = gaussian2(18, 4);

    B = A - filter2(H2, A, 'same');

    V = B .* B;

    D = filter2(H2, V, 'same');

    % Rewrite this
    E = B ./ max(sqrt(D), sqrt(mean(V(:))));

    B = E;

end

