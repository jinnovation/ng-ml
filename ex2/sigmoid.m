function g = sigmoid(z)
  g = arrayfun(@(x) 1/(1+e^(-x)), z)
end
