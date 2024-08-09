Comparison of Results:
Direct Computation time: 0.370163 seconds.
LSQR time: 0.407603 seconds.
SVD time: 22.166413 seconds.
QR time: 3.701567 seconds.

Differences between solutions:
Norm between Direct and LSQR: 8.128928e-01
Norm between Direct and SVD: 2.984406e-13
Norm between Direct and QR: 1.574757e-13
Norm between LSQR and SVD: 8.128928e-01
Norm between LSQR and QR: 8.128928e-01
Norm between SVD and QR: 2.672493e-13

Verification of Residuals:
Norm of residual for Direct Computation: 1.255158e-12
Norm of residual for LSQR: 4.180804e-01
Norm of residual for SVD: 3.685822e-12
Norm of residual for QR: 1.973699e-12




% Generate a non-square sparse matrix A (800x1000)
n = 4800;  % Number of rows
m = 4000;  % Number of columns
A = sprandn(n, m, 0.01);  % Sparse matrix with 1% non-zero entries

% Compute H = A^T * A
H = A' * A;

% Right-hand side vector b for Hx = b, the size should match H
b = randn(m, 1);  % Since H is m x m

%% Check the condition number of H
cond_H = cond(H);
fprintf('Condition number of H: %e\n', cond_H);

%% Direct Computation for Benchmark
fprintf('Running Direct Computation on H...\n');
tic;
% Directly solving Hx = b using MATLAB's backslash operator
x_direct = H \ b;
time_direct = toc;
fprintf('Direct computation completed in %f seconds.\n', time_direct);

%% LSQR Method to solve Hx = b
fprintf('Running LSQR on H...\n');
tic;

% Define an anonymous function that references the external function
Hfun = @(x, flag) afun_H(A, x, flag);

% LSQR on Hx = b, using the custom function handle
x_lsqr = lsqr(Hfun, b, 1e-6, 1000);
time_lsqr = toc;
fprintf('LSQR completed in %f seconds.\n', time_lsqr);

%% SVD Method
fprintf('Running SVD on A...\n');
tic;
% Use SVD on A to solve Hx = b
[U, S, V] = svd(full(A), 'econ');  % 'econ' for economy-sized decomposition
% Solve using the SVD components
x_svd = V * (S \ (U' * (U * (S \ (V' * b)))));
time_svd = toc;
fprintf('SVD completed in %f seconds.\n', time_svd);

%% QR Decomposition on A
fprintf('Running QR decomposition on A...\n');
tic;
% QR decomposition for Hx = b
[Q, R] = qr(A, 0);  % '0' for economy size
% Solve the system Hx = b using QR decomposition
x_qr = R \ (Q' * (Q * (R' \ b)));
time_qr = toc;
fprintf('QR decomposition completed in %f seconds.\n', time_qr);

%% Compare Results
fprintf('\nComparison of Results:\n');
fprintf('Direct Computation time: %f seconds.\n', time_direct);
fprintf('LSQR time: %f seconds.\n', time_lsqr);
fprintf('SVD time: %f seconds.\n', time_svd);
fprintf('QR time: %f seconds.\n', time_qr);

% Verify that the solutions are close to each other
fprintf('\nDifferences between solutions:\n');
fprintf('Norm between Direct and LSQR: %e\n', norm(x_direct - x_lsqr));
fprintf('Norm between Direct and SVD: %e\n', norm(x_direct - x_svd));
fprintf('Norm between Direct and QR: %e\n', norm(x_direct - x_qr));
fprintf('Norm between LSQR and SVD: %e\n', norm(x_lsqr - x_svd));
fprintf('Norm between LSQR and QR: %e\n', norm(x_lsqr - x_qr));
fprintf('Norm between SVD and QR: %e\n', norm(x_svd - x_qr));

%% Verification of Residuals
fprintf('\nVerification of Residuals:\n');
residuals = [H*x_direct - b, H*x_lsqr - b, H*x_svd - b, H*x_qr - b];
norm_direct = norm(residuals(:,1));
norm_lsqr = norm(residuals(:,2));
norm_svd = norm(residuals(:,3));
norm_qr = norm(residuals(:,4));
fprintf('Norm of residual for Direct Computation: %e\n', norm_direct);
fprintf('Norm of residual for LSQR: %e\n', norm_lsqr);
fprintf('Norm of residual for SVD: %e\n', norm_svd);
fprintf('Norm of residual for QR: %e\n', norm_qr);
