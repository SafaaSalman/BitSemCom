function y = cumsum(x, varargin)
% CUMSUM  GNU Octave stand-in that accepts MATLAB's direction flag.
%
%   Duffy's landslide.m computes the accumulated drops with cumsum(d,'reverse'), a
%   MATLAB-only call: Octave's cumsum (checked with 8.4) stops with
%   "cumsum: unrecognized string argument". This file shadows the builtin only when
%   this folder is on the Octave path, which experiments/grand_backend.py does for the
%   'octave' engine and never for MATLAB. Duffy's files are not modified.
%
%   cumsum(x, 'reverse')        -> flip(cumsum(flip(x, dim), dim), dim) along the first
%   cumsum(x, dim, 'reverse')      non-singleton dimension (MATLAB's rule) or dim
%   cumsum(x), cumsum(x, dim),  -> the builtin, unchanged
%   cumsum(..., 'native'|'double')
    reverse = false;
    args = {};
    for i = 1:numel(varargin)
        a = varargin{i};
        if ischar(a) && strcmpi(a, 'reverse')
            reverse = true;
        elseif ischar(a) && strcmpi(a, 'forward')
            % MATLAB default direction, nothing to do
        else
            args{end + 1} = a;  %#ok<AGROW>
        end
    end
    if ~reverse
        y = builtin('cumsum', x, args{:});
        return;
    end
    if ~isempty(args) && isnumeric(args{1})
        dim = args{1};
        rest = args(2:end);
    else
        dim = find(size(x) ~= 1, 1);
        if isempty(dim)
            dim = 1;
        end
        rest = args;
    end
    y = flip(builtin('cumsum', flip(x, dim), dim, rest{:}), dim);
end
