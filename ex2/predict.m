function p = predict(theta, X)
%predict predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = predict(theta, x) computes the predictions for x using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

  p = arrayfun(@(r) r > 0.5, sigmoid(X * theta))
end
