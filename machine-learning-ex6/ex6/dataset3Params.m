function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;


C_choices=[0.01, 0.03, 0.1, 0.3, 1, 3,10,30 ];
Sigma_choices=[0.01, 0.03, 0.1, 0.3, 1, 3,10,30 ];

result_matrix = zeros(length(C_choices)^2,3);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

iter = 1;

for i = 1: length(C_choices)
    for j = 1:length(Sigma_choices)
    curr_c = C_choices(i);
    curr_sigma = Sigma_choices(j);
    model= svmTrain(X, y, curr_c, @(x1, x2) gaussianKernel(x1, x2, curr_sigma));
    pred = svmPredict(model, Xval);
    
    
    error = mean(double(pred ~= yval));
    
    result_matrix(iter, :) = [curr_c, curr_sigma, error];
    iter = iter + 1;    
    end
    
end    

% size(result_matrix)
% result_matrix
% pause;

[V,I] = min(result_matrix(:,3));
params = result_matrix(I,:);

C = params(1);
sigma = params(2);






% =========================================================================

end
