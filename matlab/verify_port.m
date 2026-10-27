function ok = verify_port(testset_file)
% Compare Duffy's MATLAB ORBGRAND with the Python port on an exported test set.
%
%   >> cd matlab
%   >> addpath('GRAND-MATLAB')        % wherever bin_ORBGRAND1.m and landslide.m live
%   >> verify_port('../results/grand_verify/testset.mat')
%
% The test set is written by `python experiments/grand_verify.py export`. The decisions
% of the reference are also saved next to it (matlab_out.mat) so that
% `python experiments/grand_verify.py compare` can print the same report.
    if nargin < 1
        testset_file = fullfile('..', 'results', 'grand_verify', 'testset.mat');
    end
    S = load(testset_file);
    variant = strtrim(char(S.variant));
    fprintf('code %s, variant %s, budget %d, %d blocks\n', strtrim(char(S.code)), variant, S.max_queries, size(S.llr, 1));
    [words, guesses, abandoned] = orbgrand_decode_blocks(double(S.H), double(S.max_queries), variant, double(S.llr));

    wp = round(S.words_py); gp = round(S.guesses_py(:)); ap = round(S.abandoned_py(:)) ~= 0;
    am = abandoned(:) ~= 0;
    both = ~ap & ~am;
    word_mis = sum(any(wp(both, :) ~= words(both, :), 2));
    aband_mis = sum(ap ~= am);
    guess_mis = sum(gp ~= guesses(:));
    fprintf('abandonment mismatches : %d\n', aband_mis);
    fprintf('guess-count mismatches : %d\n', guess_mis);
    fprintf('codeword mismatches    : %d of %d decoded by both\n', word_mis, sum(both));
    ok = (word_mis == 0) && (aband_mis == 0) && (guess_mis == 0);
    if ok
        fprintf('RESULT: IDENTICAL - the Python port reproduces the MATLAB reference bit for bit\n');
    else
        fprintf('RESULT: MISMATCH - inspect\n');
    end
    [d, ~, ~] = fileparts(testset_file);
    backend = 'matlab';
    save(fullfile(d, 'matlab_out.mat'), 'words', 'guesses', 'abandoned', 'backend', '-v7');
end
