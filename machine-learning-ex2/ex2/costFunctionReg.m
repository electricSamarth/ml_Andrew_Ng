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

h_raw=((theta)'*(X)')';
h_actual=1./((e.^(-h_raw))+1);
jc1=-(y.*log(h_actual));
jc2=(1-y).*log(1-h_actual);
t_sum_j=theta.^2;
t_sum_j(1)=0;
lc=lambda/(2*m);
J=((sum(jc1-jc2))/m)+((lc*(sum(t_sum_j))));
grad=((((h_actual-y)'*X)')/m)+((lambda*(theta))/m);
grad(1)=((((h_actual-y)'*X)')/m)(1);






% =============================================================

end
