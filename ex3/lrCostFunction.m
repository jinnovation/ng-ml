function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

 
  ## the below copied from ex2

m = length(y); % number of training examples
h = sigmoid(X * theta);

J = mean(-y .* log(h) - (1-y) .* log(1-h)) + (lambda/(2*m)) * sum(theta(2:end).^2);
Xnobias = X(:,2:end);
grad = mean(repmat(h-y, [1 size(Xnobias)(2)]) .* Xnobias)...
       + transpose((lambda/m) * theta(2:end,:));
grad = transpose([ mean((h-y) .* X(:,1)) grad ]);

end
