function orbgrand_decode_file(in_file, out_file)
% Decode a batch of LLR blocks with Ken Duffy's reference ORBGRAND and save the result.
%
%   orbgrand_decode_file('in.mat', 'out.mat')
%
% in.mat  : llr (nb x n, positive = bit 0), H ((n-k) x n), max_queries, variant ('1line'|'basic')
% out.mat : words (nb x n; -1 rows for abandoned blocks), guesses (nb x 1), abandoned (nb x 1)
%
% Requires bin_ORBGRAND1.m / bin_ORBGRAND.m / landslide.m from
% https://github.com/kenrduffy/GRAND-MATLAB on the path (see README.md).
% Runs unchanged in MATLAB and in GNU Octave.
    S = load(in_file);
    llr = double(S.llr);
    H = double(S.H);
    max_queries = double(S.max_queries);
    variant = strtrim(char(S.variant));
    [nb, n] = size(llr);
    [words, guesses, abandoned] = orbgrand_decode_blocks(H, max_queries, variant, llr);
    fprintf('decoded %d blocks of n=%d: %d abandoned, mean %.1f guesses\n', nb, n, sum(abandoned), mean(guesses));
    save(out_file, 'words', 'guesses', 'abandoned', '-v7');
end
