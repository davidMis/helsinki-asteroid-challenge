function hac_evaluate_public(requestPath)
% Native driver: call the externally supplied twoDmetric.m without edits.
request = jsondecode(fileread(requestPath));
addpath(request.source_dir, '-begin');
cd(request.source_dir);
assert(strcmp(which('twoDmetric'), fullfile(request.source_dir, 'twoDmetric.m')), ...
    'HAC:WrongSource', 'MATLAB resolved an unexpected twoDmetric.m.');
required = {'stlread', 'poly2mask', 'imfill', 'bwareafilt', 'bwboundaries', 'pdist2'};
for k = 1:numel(required)
    assert(~isempty(which(required{k})), 'HAC:MissingDependency', ...
        ['Missing MATLAB function ' required{k} '. Install Image Processing ' ...
         'Toolbox and Statistics and Machine Learning Toolbox.']);
end
runtime = struct('version', version, 'release', version('-release'), ...
    'computer', computer, 'twoDmetric_path', which('twoDmetric'), ...
    'installed_products', ver);
result = struct('runtime', runtime, 'rows', {{}});
for k = 1:numel(request.jobs)
    job = request.jobs(k);
    timer = tic;
    row = struct('model_id', job.model_id, 'theta', job.theta, ...
        'status', 'ok', 'projection_score', 0, 'elapsed_seconds', 0, 'error', '');
    try
        row.projection_score = twoDmetric(job.target_stl, job.candidate_stl, job.theta);
        assert(isfinite(row.projection_score) && row.projection_score >= 0 && ...
            row.projection_score <= 1, 'HAC:InvalidScore', 'Invalid projection score.');
    catch exception
        row.status = 'error';
        row.error = getReport(exception, 'extended', 'hyperlinks', 'off');
    end
    row.elapsed_seconds = toc(timer);
    result.rows{end + 1} = row;
    temporary = [request.result_path '.tmp'];
    fid = fopen(temporary, 'w');
    assert(fid >= 0, 'HAC:OutputFailure', 'Cannot write MATLAB result.');
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '%s', jsonencode(result));
    clear cleaner;
    movefile(temporary, request.result_path, 'f');
    fprintf('HAC Model %02d: %s, score %.12g (%.3f seconds)\n', ...
        job.model_id, row.status, row.projection_score, row.elapsed_seconds);
    assert(strcmp(row.status, 'ok'), 'HAC:EvaluationFailure', row.error);
end
end
