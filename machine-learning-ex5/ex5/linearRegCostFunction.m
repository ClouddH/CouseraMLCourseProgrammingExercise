function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% cost function 

% size(X)
% size(theta)
% size(y)

J = (1/ (2*m) )*  sum( ((X * theta) -y).^2)  + (lambda/(2 * m)) * sum( theta([2:end ],:).^2 );



% grad
% size(((X*theta) -y))
% size( X(:, 1))
% size(X(:, [2:end])')
% size(theta([2:end],:))
% size(X)
% size((1/m) * sum(((X*theta) -y)* X(:, [2:end])'))
% size((lambda /m) * theta([2:end], :))
% size(((X*theta) -y))
% size(X)
% size(theta)
% pause;
% 
% grad = [ (1/m)* sum(((X*theta) -y).* X(:, 1)), (1/m) * sum(((X*theta) -y)* X(:, [2:end])')  + (lambda /m) * theta([2:end], :)];

% dim(X) (12, 2)  dim(y) (12 1)  dim(theta) (2,1)



% size(X(:,1)' * (X*theta -y)) % one number
% size(( X(:, [2:end])'* (X*theta - y)))  % dim(2,1)
% pause



grad = [(1/m) * ( X(:,1)' * (X * theta -y));     ((1/m) * ( X(:, [2:end])' * (X*theta - y))) + ((lambda/ m) * theta([2:end] ,:))];

% =========================================================================

grad = grad(:);

end
