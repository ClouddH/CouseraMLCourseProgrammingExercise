function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

positive_index =  y(:,1) ==1;
negative_index =  y(:,1) ==0;

positive_data = X(positive_index,:);
negative_data = X(negative_index,:);


plot(positive_data(:,1),positive_data(:,2),'k+');
plot(negative_data(:,1),negative_data(:,2),'ko','MarkerFaceColor','y');
% disp(positive_data);
% disp(negative_data);


% =========================================================================



hold off;

end
