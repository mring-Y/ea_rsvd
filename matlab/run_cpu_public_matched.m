function run_cpu_public_matched(output_csv, data_root)
% Reproduce the CPU rSVD matched-width quality check for public non-LP inputs.
% Protocol: k=16, sketch width ell=24, q=1, five fixed seeds.
repo_root = fileparts(fileparts(mfilename('fullpath')));
if nargin < 1 || isempty(output_csv)
    output_csv = fullfile(repo_root, 'results', 'matlab', 'cpu_public_matched.csv');
end
if nargin < 2 || isempty(data_root)
    data_root = fullfile(repo_root, 'data', 'suitesparse');
end

names = {'ca-AstroPh', 'ca-HepPh', 'email-EuAll', 'p2p-Gnutella31'};
seeds = 20261003:20261007;
k = 16; ell = 24; q = 1;
out_dir = fileparts(output_csv);
if ~isempty(out_dir) && ~isfolder(out_dir), mkdir(out_dir); end
fid = fopen(output_csv, 'w');
if fid < 0, error('Cannot open output CSV: %s', output_csv); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, ['matrix,rows,cols,nnz,seed,rank,sketch_width,power_depth,', ...
    'cpu_rsvd_ms,rank_k_residual,projection_residual,q_orthogonality,validation\n']);

for mi = 1:numel(names)
    name = names{mi};
    path = resolve_matrix_path(data_root, name);
    if isempty(path), warning('Missing %s under %s', name, data_root); continue; end
    A = read_matrix_market_local(path);
    [m, n] = size(A);
    normA2 = full(sum(sum(A .* A)));
    fprintf('CPU matched %s (%dx%d, nnz=%d)\n', name, m, n, nnz(A));
    for seed = seeds
        rng(seed, 'twister');
        t0 = tic;
        Omega = randn(n, ell);
        [Q, ~] = qr(A * Omega, 0);
        for it = 1:q
            [Z, ~] = qr(A' * Q, 0);
            [Q, ~] = qr(A * Z, 0);
        end
        B = Q' * A;
        sigma = svd(full(B), 'econ');
        elapsed_ms = toc(t0) * 1000;
        projection_residual = sqrt(max(normA2 - sum(sigma.^2), 0) / normA2);
        rank_k_residual = sqrt(max(normA2 - sum(sigma(1:k).^2), 0) / normA2);
        q_orth = norm(Q' * Q - eye(size(Q, 2)), 'fro');
        pass = isfinite(rank_k_residual) && isfinite(projection_residual) && q_orth < 1e-8;
        fprintf(fid, '%s,%d,%d,%d,%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%s\n', ...
            name, m, n, nnz(A), seed, k, ell, q, elapsed_ms, ...
            rank_k_residual, projection_residual, q_orth, ternary(pass, 'PASS', 'FAIL'));
    end
end
end

function path = resolve_matrix_path(data_root, name)
candidates = { ...
    fullfile(data_root, name, name, [name '.mtx']), ...
    fullfile(data_root, name, [name '.mtx'])};
path = '';
for i = 1:numel(candidates)
    if isfile(candidates{i}), path = candidates{i}; return; end
end
end

function out = ternary(condition, yes, no)
if condition, out = yes; else, out = no; end
end

function A = read_matrix_market_local(path)
fid = fopen(path, 'r');
if fid < 0, error('Cannot open %s', path); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
header = fgetl(fid);
if ~ischar(header) || ~startsWith(lower(header), '%%matrixmarket matrix coordinate')
    error('Unsupported Matrix Market header in %s', path);
end
is_pattern = contains(lower(header), 'pattern');
line = fgetl(fid);
while ischar(line) && (isempty(strtrim(line)) || startsWith(strtrim(line), '%'))
    line = fgetl(fid);
end
dims = sscanf(line, '%d %d %d');
if numel(dims) ~= 3, error('Invalid Matrix Market dimensions in %s', path); end
raw = textscan(fid, '%d %d %f', dims(3), 'CommentStyle', '%');
ii = raw{1}; jj = raw{2};
if is_pattern, vv = ones(size(ii)); else, vv = raw{3}; end
A = sparse(ii, jj, vv, dims(1), dims(2));
end
