function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

h = sigmoid(X * theta)
J = mean(-y .* log(h) - (1-y) .* log(1-h))
grad = mean(repmat((h-y), [1 size(X)(2)]) .* X)
end
