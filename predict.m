
function predict(weight_path, test_data_path)
  [y, X] = libsvmread(test_data_path);
  load(weight_path);

  % since kddb.t is the sparse representation, many features are missing
  w = w(1:size(X, 2));

  predictions = (1 ./ (1+exp(-X * w))) > 0.5;
  acc = nnz(~(y - predictions)) ./ length(y) * 100;
  fprintf('accuracy: %f %%\n', acc);
end
