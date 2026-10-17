function generate_paper_assets()
% Generate figures and controlled-spectrum data used by the manuscript.
% Set EA_WORK_ROOT to the directory containing review_experiments and the
% platform result folders. Set EA_OUTPUT_ROOT to choose the output folder.

root = getenv('EA_WORK_ROOT');
if isempty(root)
    root = pwd;
end
out = getenv('EA_OUTPUT_ROOT');
if isempty(out)
    out = fullfile(root, 'paper_assets');
end
if ~exist(out, 'dir')
    mkdir(out);
end

set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 10);
set(groot, 'defaultLineLineWidth', 1.25);

rankDir = fullfile(root, 'review_experiments', 'round20_rank_adequacy_multiseed');
fixedDir = fullfile(root, 'review_experiments', 'round16_exact_fixed_grid');
precisionFile = fullfile(root, 'autodl_cloud', 'local_precision_replay_20261003_fixed', 'precision_comparison.csv');
a800File = resolve_result_file(root, 'EA_A800_CSV', '_a800_results_*');
gpu4090File = resolve_result_file(root, 'EA_RTX4090D_CSV', 'autodl_results_*');

rankGrid = readtable(fullfile(rankDir, 'rank_adequacy_grid.csv'));
rankSummary = readtable(fullfile(rankDir, 'rank_adequacy_summary.csv'));
fixedGrid = readtable(fullfile(fixedDir, 'fixed_grid.csv'));

% Figure 1: all rank-sweep states in the diagnostic plane.
fig = figure('Color', 'w', 'Position', [80 80 900 620]);
hold on;
mats = unique(string(rankGrid.matrix), 'stable');
colors = lines(numel(mats));
for i = 1:numel(mats)
    mask = string(rankGrid.matrix) == mats(i);
    scatter(rankGrid.exact_projection_residual(mask), rankGrid.exact_rank_residual(mask), 18, ...
        'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', 'none', 'DisplayName', char(mats(i)));
end
plot([0 1], [0 1], 'k:', 'DisplayName', 'rho_k = rho_l');
xline(0.20, 'k--', 'HandleVisibility', 'off');
yline(0.20, 'k--', 'HandleVisibility', 'off');
xlabel('Projection residual, rho_l');
ylabel('Rank-k residual of computed projected factor, rho_k');
title('Residual-pair diagnosis across 925 five-seed states');
xlim([0 1.02]); ylim([0 1.02]); grid on; axis square;
legend('Location', 'southeast', 'NumColumns', 2);
text(0.015, 0.965, 'both residuals below 0.20', 'FontSize', 9);
text(0.015, 0.52, 'rank-limited', 'FontSize', 9);
text(0.48, 0.965, 'projection-limited', 'FontSize', 9);
exportgraphics(fig, fullfile(out, 'fig1_residual_diagnostic.png'), 'Resolution', 300);
close(fig);

% Figure 2: target-rank curves with five-seed confidence intervals.
fig = figure('Color', 'w', 'Position', [80 80 980 620]);
hold on;
for i = 1:numel(mats)
    mask = string(rankSummary.matrix) == mats(i);
    rows = rankSummary(mask, :);
    rows = sortrows(rows, 'rank');
    errorbar(rows.rank, rows.best_rho_k_mean, rows.best_rho_k_ci95_halfwidth, '-o', ...
        'Color', colors(i,:), 'MarkerFaceColor', colors(i,:), 'DisplayName', char(mats(i)));
end
yline(0.20, 'k--', 'target threshold');
xlabel('Requested rank k'); ylabel('Best tested rho_k');
title('Rank adequacy changes across matrix families');
xticks([8 16 32 64]); xlim([6 66]); ylim([0 1.02]); grid on;
legend('Location', 'eastoutside');
exportgraphics(fig, fullfile(out, 'fig2_rank_curves.png'), 'Resolution', 300);
close(fig);

% Figure 3: fixed-grid quality/cost frontier.
fig = figure('Color', 'w', 'Position', [60 60 1050 650]);
hold on;
for i = 1:numel(unique(string(fixedGrid.matrix), 'stable'))
    fmats = unique(string(fixedGrid.matrix), 'stable');
    mask = string(fixedGrid.matrix) == fmats(i);
    scatter(fixedGrid.gpu_pipeline_ms(mask), fixedGrid.exact_rank_residual(mask), 13, ...
        'MarkerFaceColor', colors(mod(i-1,size(colors,1))+1,:), 'MarkerEdgeColor', 'none', ...
        'DisplayName', char(fmats(i)));
    [~, idx] = min(fixedGrid.exact_rank_residual(mask));
    vals = find(mask);
    plot(fixedGrid.gpu_pipeline_ms(vals(idx)), fixedGrid.exact_rank_residual(vals(idx)), ...
        'kp', 'MarkerSize', 9, 'MarkerFaceColor', 'y', 'HandleVisibility', 'off');
end
xlabel('GPU pipeline time (ms)'); ylabel('Computed projected-factor rho_k');
title('Fixed-grid quality/cost frontier (450 states)');
set(gca, 'YScale', 'log'); grid on;
legend('Location', 'eastoutside', 'NumColumns', 2);
exportgraphics(fig, fullfile(out, 'fig3_fixed_grid_frontier.png'), 'Resolution', 300);
close(fig);

% Figure 4: stage timing on the two available GPU platforms.
if isfile(a800File) && isfile(gpu4090File)
    a800 = readtable(a800File);
    gpu4090 = readtable(gpu4090File);
    common = intersect(string(a800.input), string(gpu4090.input), 'stable');
    % Keep the ten matrices present in both records and sort by 4090D time.
    [~, order] = sort(gpu4090.gpu_pipeline_ms(ismember(string(gpu4090.input), common)), 'descend');
    common = common(order);
    if numel(common) > 10
        common = common(1:10);
    end
    fig = figure('Color', 'w', 'Position', [60 60 1100 700]);
    tiledlayout(2, 1, 'TileSpacing', 'compact');
    stages = {'h2d_ms','spmm_forward_ms','spmm_transpose_ms','qr_ms','svd_ms'};
    labels = {'H2D','SpMM A*Omega','SpMM A^T*Q','QR','SVD'};
    for p = 1:2
        nexttile; hold on;
        if p == 1
            D = gpu4090; name = 'RTX 4090D';
        else
            D = a800; name = 'A800';
        end
        vals = zeros(numel(common), numel(stages));
        for j = 1:numel(common)
            row = find(string(D.input) == common(j), 1);
            for s = 1:numel(stages)
                vals(j,s) = D{row, stages{s}};
            end
        end
        bar(vals, 'stacked');
        ylabel('Milliseconds'); title([name ' stage timing']);
        xticks(1:numel(common)); xticklabels(cellstr(short_names(common))); xtickangle(35);
        grid on; if p == 1, legend(labels, 'Location', 'eastoutside'); end
    end
    exportgraphics(fig, fullfile(out, 'fig4_cross_platform_stages.png'), 'Resolution', 300);
    close(fig);
end

% Figure 5: precision replay consistency.
if isfile(precisionFile)
    precision = readtable(precisionFile, 'FileType', 'text');
    fig = figure('Color', 'w', 'Position', [80 80 950 620]);
    tiledlayout(1, 2, 'TileSpacing', 'compact');
    matrices = string(precision.matrix);
    nexttile; hold on;
    um = unique(matrices, 'stable');
    maxL = zeros(numel(um),1); maxK = zeros(numel(um),1);
    for i = 1:numel(um)
        mask = matrices == um(i);
        maxL(i) = max(precision.max_abs_rho_l_diff(mask));
        maxK(i) = max(precision.max_abs_rho_k_diff(mask));
    end
    bar([maxL maxK]); set(gca, 'YScale', 'log');
    xticks(1:numel(um)); xticklabels(cellstr(short_names(um))); xtickangle(35);
    ylabel('Maximum absolute difference'); title('Precision replay residual differences');
    legend({'rho_l','rho_k'}, 'Location', 'northwest'); grid on;
    nexttile; 
    disagreements = precision.mode_decision_disagreements;
    qerr = precision.max_q_orth;
    bar(categorical(cellstr(short_names(um))), [disagreements; qerr].');
    ylabel('Count / maximum error'); title('Decision agreement and Q orthogonality');
    legend({'decision disagreements','max Q error'}, 'Location', 'northwest'); grid on;
    exportgraphics(fig, fullfile(out, 'fig5_precision_replay.png'), 'Resolution', 300);
    close(fig);
end

% Controlled spectra: CPU double-precision randomized range finder.
controlled = controlled_spectrum_experiment();
writetable(controlled, fullfile(out, 'controlled_spectrum_diagnosis.csv'));
fig = figure('Color', 'w', 'Position', [80 80 1000 560]);
tiledlayout(1, 2, 'TileSpacing', 'compact');
families = unique(string(controlled.family), 'stable');
for i = 1:numel(families)
    nexttile; hold on;
    mask = string(controlled.family) == families(i);
    ranks = unique(controlled.target_rank(mask));
    cc = lines(numel(ranks));
    for j = 1:numel(ranks)
        mm = mask & controlled.target_rank == ranks(j);
        scatter(controlled.rho_l(mm), controlled.rho_k(mm), 32, cc(j,:), 'filled', ...
            'DisplayName', sprintf('k=%d', ranks(j)));
    end
    plot([0 1],[0 1],'k:'); xline(0.20,'k--'); yline(0.20,'k--');
    xlim([0 1.02]); ylim([0 1.02]); axis square; grid on;
    xlabel('rho_l'); ylabel('rho_k'); title(char(families(i)));
    legend('Location','southeast');
end
exportgraphics(fig, fullfile(out, 'fig6_controlled_spectra.png'), 'Resolution', 300);
close(fig);

summary = table();
summary.metric = ["rank_sweep_states"; "fixed_grid_states"; "rank_transition_max_abs_delta_rho_l"; ...
    "precision_decision_disagreements"; "controlled_spectrum_rows"];
summary.value = [height(rankGrid); height(fixedGrid); max(readtable(fullfile(rankDir, 'rank_transition_by_seed.csv')).abs_delta_rho_l); ...
    sum(readtable(precisionFile).mode_decision_disagreements); height(controlled)];
writetable(summary, fullfile(out, 'paper_asset_summary.csv'));

fprintf('Generated paper assets in %s\n', out);
end

function names = short_names(paths)
names = strings(size(paths));
for i = 1:numel(paths)
    token = regexp(char(paths(i)), '[\\/](?:[^\\/]+[\\/])?([^\\/]+)\\.mtx$', 'tokens', 'once');
    if isempty(token)
        token = regexp(char(paths(i)), '([^\\/]+)\\.mtx$', 'tokens', 'once');
    end
    if isempty(token), names(i) = paths(i); else, names(i) = string(token{1}); end
end
end

function path = resolve_result_file(root, variable, folder_pattern)
path = getenv(variable);
if ~isempty(path)
    if ~isfile(path), error('%s does not exist: %s', variable, path); end
    return;
end
candidates = dir(fullfile(root, folder_pattern, 'same_cuda_sparse_reference.csv'));
if isempty(candidates)
    path = fullfile(root, folder_pattern, 'same_cuda_sparse_reference.csv');
    return;
end
[~, newest] = max([candidates.datenum]);
path = fullfile(candidates(newest).folder, candidates(newest).name);
end

function T = controlled_spectrum_experiment()
rng(20261010, 'twister');
m = 256; n = 192; seeds = 1:5;
families = {'rank-limited spectrum', 'projection-limited spectrum'};
rows = [];
for f = 1:2
    if f == 1
        s = [linspace(1, 0.80, 32), zeros(1, n-32)];
        widths = [40 64];
    else
        s = 1 ./ sqrt(1:n);
        widths = [8 16 32];
    end
    [U,~] = qr(randn(m,m), 0); [V,~] = qr(randn(n,n), 0);
    A = U(:,1:n) * diag(s) * V';
    energy = norm(A, 'fro')^2;
    for seed = seeds
        rng(20261010 + seed, 'twister');
        for l = widths
            for q = 0:2
                Omega = randn(n,l);
                Y = A * Omega;
                Q = orth(Y);
                for step = 1:q
                    Q = orth(A' * Q);
                    Q = orth(A * Q);
                end
                B = Q' * A;
                sb = svd(B, 'econ');
                for k = [8 16 32]
                    kk = min(k, numel(sb));
                    rhoL = sqrt(max(energy - norm(B, 'fro')^2, 0) / energy);
                    rhoK = sqrt(max(energy - sum(sb(1:kk).^2), 0) / energy);
                    rows = [rows; {families{f}, seed, l, q, k, rhoL, rhoK}]; %#ok<AGROW>
                end
            end
        end
    end
end
T = cell2table(rows, 'VariableNames', {'family','seed','sketch_width','power_steps','target_rank','rho_l','rho_k'});
end
