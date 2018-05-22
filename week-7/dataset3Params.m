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



parameters = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
parametersLength = size(parameters, 2);
minError = 9 * 10^100; % Just a large number

for i = 1:parametersLength
  CCurrent = parameters(i);

  for j = 1:parametersLength
    sigmaCurrent = parameters(j);
    model = svmTrain(X, y, CCurrent, @(X, y) gaussianKernel(X, y, sigmaCurrent));
    error = mean(double(svmPredict(model, Xval) ~= yval));

    if (error < minError)
      minError = error;
      C = CCurrent;
      sigma = sigmaCurrent;
    endif
  endfor
endfor



% =========================================================================

end
