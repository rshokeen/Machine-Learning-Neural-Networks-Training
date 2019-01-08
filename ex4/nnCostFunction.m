function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); %provides 25 by 401 matrix

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); %provides 10 by 26 matrix
%Theta2
%size(Theta2)
% Setup some useful variables
m = size(X, 1);

 
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); %same as Theta1 size
Theta2_grad = zeros(size(Theta2)); %same as Theta2 size

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% Add ones to the X data matrix
X = [ones(m, 1) X];
a2 = sigmoid(X * Theta1');

% Add ones to the a2 data matrix
a2 = [ones(m, 1) a2];
a3 = sigmoid(a2 * Theta2');


%create logical vector of y values for 5000 training examples i.e. 5000 by 10 matrix
yv = eye(num_labels)(y,:);


%Add the cost of each of the output nos.

  %without regularization
  J = sum((- yv .* log(a3)) - ((1 - yv) .* log(1 - a3)))/m;
  J = sum(J);
   
   
  %with regularization
  %JReg = (sum((Theta1(2:end).^2)) + sum((Theta2(2:end).^2))) * lambda/(2*m);
  JReg = (sum((Theta1(1:end, 2:end)(:).^2)) + sum((Theta2(1:end, 2:end)(:).^2))) * lambda/(2*m);
  J = J + JReg;
 
  
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

for t = 1:m
  
  % No need to Add ones to the X data matrix as its already done above
  %X
  %X(1,:)
  a1 = X(t,:); %get individual training example row
  a2 = sigmoid(a1 * Theta1'); %provides each unit's output in layer 2
  a2 = [ones(1, 1) a2]; %add 1 bias unit to a2
  a3 = sigmoid(a2 * Theta2'); %provides output for each unit in last layer
  
  yv = eye(num_labels)(y,:); %create logical vector of actual y values
  delta3 = a3 - yv(t,:); %error at output layer
  
  %error at hidden layer
  %delta2 = (Theta2' * delta3') .* sigmoidGradient(a2');
  delta2 = (Theta2' * delta3');
  delta2 = delta2(2:end); %remove bias term from delta2
  delta2 = delta2 .* sigmoidGradient(a1 * Theta1')'; %z2 = a1 * Theta1'
  
  %calculate theta gradient for hidden layer to accumulate partial derivatives
  Theta2_grad = Theta2_grad + (delta3' * a2);
  
  %calculate theta gradient for 1st layer to accumulate partial derivatives
  Theta1_grad = Theta1_grad + (delta2 * a1);
  
end

%Obtain the (unregularized) gradient for the neural network cost function by dividing the accumulated gradients by 1/m
Theta2_grad = (1/m) .* Theta2_grad;
Theta1_grad = (1/m) .* Theta1_grad;

  

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);

  














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
