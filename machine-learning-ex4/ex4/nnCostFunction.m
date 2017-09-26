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
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% compute h(x)
X = [ones(m, 1), X];
z2 = X * Theta1';

a2 = sigmoid(z2);

a2_size = size(a2,1);

a2 = [ones(a2_size,1), a2];

output = sigmoid(a2 * Theta2');




% convert y from numerical label to 10 dimensional vector 

size_y = size(y,1);



converted_y = zeros(size_y, num_labels);



for i= 1: size_y
   
   converted_y(i, y(i)) = 1;
    
end



% translate cost func formula

error_accumulator = 0;

for i = 1:m
    y_of_i = converted_y(i, :);   
    h_of_x_of_i = output(i, :);  
    
    error_accumulator = error_accumulator + sum((-y_of_i * log( h_of_x_of_i' )) -  ( (1 - y_of_i )* log((1 - h_of_x_of_i)' )));
    
end

Theta1_square_sans_bias_terms = (Theta1(:, [2:end])).^2;
Theta2_square_sans_bias_terms = (Theta2(:, [2:end])).^2;

regularized_term = (lambda/(2*m)) * (sum(Theta1_square_sans_bias_terms(:)) + sum(Theta2_square_sans_bias_terms(:)));
J = ( (1/m) * error_accumulator ) + regularized_term;



% Incorrect vectorized implementation
% Why is the math wrong ?  See cousera forum 
% J = (1 / m) * sum(log(output) * (-1 * converted_y')) - (log(1- output ) * (1 - converted_y )' );



% Bakc-prop implementation
% size(Theta1_grad)
% size(Theta2_grad)
% pause
% dim(theta2_grad) = (10 26)

for i = 1:m
   delta_l3_of_i =  (output(i,:) - converted_y(i, :))';

   Theta2_grad = Theta2_grad +  (delta_l3_of_i * a2(i,:));
 
   
   delta_l2_of_i =   (Theta2' * delta_l3_of_i) .* sigmoidGradient( [1, z2(i, :)]');
   delta_l2_of_i =delta_l2_of_i(2:end);
   
   Theta1_grad = Theta1_grad +  (delta_l2_of_i * X(i,:));
  
end



Theta1_grad = [ (1/m).*Theta1_grad(:,1), (1/m).*Theta1_grad(:,[2:end]) + (lambda / m) * (Theta1(:,[2:end])) ];
Theta2_grad = [ (1/m).*Theta2_grad(:,1), (1/m).*Theta2_grad(:,[2:end]) + (lambda / m) * (Theta2(:,[2:end])) ];





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
