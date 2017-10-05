function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

errorSummation = 0;
gradSummation  = zeros(size(theta));
for i = 1:m
  % Hypothesis function result
  h = sigmoid(theta' * X(i, :)');

  errorSummation += -y(i) * log(h) - (1 - y(i)) * log(1 - h);

  for j = 1:size(theta)
    gradSummation(j) += (h - y(i)) * X(i, j);
  end
end

regSummation = 0;

grad = gradSummation ./ m;

% Should not regularize the first term
for j = 2:size(theta)
  regSummation += theta(j) ^ 2;

  % Regularize for the gradient descendent part
  grad(j) += lambda / m * theta(j);
end

% Calculate the cost function's value
J = (1 / m) * errorSummation + lambda / (2 * m) * regSummation;

% =============================================================

end
