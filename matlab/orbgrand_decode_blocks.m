function [words, guesses, abandoned] = orbgrand_decode_blocks(H, max_queries, variant, llr)
% Loop over LLR rows and call Duffy's decoder on each one (no modification of his code).
%   variant '1line' -> bin_ORBGRAND1 (default), 'basic' -> bin_ORBGRAND
    [nb, n] = size(llr);
    words = -ones(nb, n);
    guesses = zeros(nb, 1);
    abandoned = zeros(nb, 1);
    use_basic = strcmp(variant, 'basic');
    t0 = tic;
    for i = 1:nb
        if use_basic
            [y, ~, g, a] = bin_ORBGRAND(H, max_queries, llr(i, :));
        else
            [y, ~, g, a] = bin_ORBGRAND1(H, max_queries, llr(i, :));
        end
        words(i, :) = double(y);
        guesses(i) = g;
        abandoned(i) = a;
        if mod(i, 100) == 0
            fprintf('  %d/%d blocks (%.0fs)\n', i, nb, toc(t0));
        end
    end
end
