function y = sigmoid(X, theta)

z = X * theta;
y = z;
y = 1 ./ (1 + (e .^ -z));
