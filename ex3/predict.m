function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

layer1 = sigmoid([ones(size(X,1),1) X] * transpose(Theta1));
layer2 = sigmoid([ones(size(layer1, 1), 1) layer1] * transpose(Theta2));
layer3 = sigmoid(layer2);

[_, p] = max(layer3, [],2);
end
