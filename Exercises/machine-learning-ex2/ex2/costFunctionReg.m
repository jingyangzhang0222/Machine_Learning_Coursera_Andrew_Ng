function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

m = length(y); % number of training examples
n = length(theta);
% You need to return the following variables correctly 
prediction = sigmoid(X * theta);
J = sum((-y).*log(prediction) - (1 - y).*(log(1 - prediction))) / m + (lambda / (2 * m)) * sum(theta(2:n)' * theta(2:n));
grad = zeros(size(theta));
for j = 1:n
    for i = 1:m
        if j == 1
            grad(j) = grad(j) + (prediction(i) - y(i)) .* X(i, j);
        else
            grad(j) = grad(j) + (prediction(i) - y(i)) .* X(i, j) + lambda / m * theta(j);
        end
    end
end
grad = grad / m;




% =============================================================

end
