function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
% You need to return the following variables correctly 
prediction = sigmoid(X * theta);
J = sum((-y).*log(prediction) - (1 - y).*(log(1 - prediction)));
J = J / m;
grad = zeros(size(theta));
for j = 1:n
    for i = 1:m
        grad(j) = grad(j) + (prediction(i) - y(i)) .* X(i, j);
    end
end
grad = grad / m;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
