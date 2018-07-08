
function train(data_path, mode, C)
  tic
  [y, X] = libsvmread(data_path);
  y = y * 2 - 1;

  fprintf('data loaded\n');
  maxNumCompThreads(32);
  eta = 0.01; xi = 0.1;

  w = lr_gradient_descent(mode, y, X, eta, C, xi);
  toc
  save('weights.mat', 'w');
end

function w = lr_gradient_descent(mode, y, X, eta, C, xi)
  epsilon = 0.01;
  w = zeros(size(X, 2), 1);
  last_grad = 0;
  exp_neg_ywx = exp(-y .* (X * w));

  grad = lr_gradient(w, C, exp_neg_ywx, y, X);
  grad_norm_threshold = epsilon * norm(grad);

  for i = 1:100
    grad_norm = norm(grad);
    fprintf('|g|: %f\n', grad_norm);
    if grad_norm < grad_norm_threshold
      fprintf('loss: %f\n', cost_func(w, C, exp_neg_ywx));
      break;
    end

    if mode == 'gd'
      s = -grad;
    else
      s = conjugate_solve(xi, grad, grad_norm, X, exp_neg_ywx, C);
    end

    [w, exp_neg_ywx] = line_search(w, C, y, X, exp_neg_ywx, eta, grad, s);
    grad = lr_gradient(w, C, exp_neg_ywx, y, X);
  end
end

function g = lr_gradient(w, C, exp_neg_ywx, y, X)
  g = w + (C * ((1 ./ (1 + exp_neg_ywx) - 1) .* y)' * X)';
end

function cost = cost_func(w, C, exp_neg_ywx)
  cost = 0.5 * w' * w + C * sum(log(1 + exp_neg_ywx));
end

function [w, exp_neg_ywx] = line_search(w, C, y, X, exp_neg_ywx, eta, grad, s)
  alpha = 1;
  f_w = cost_func(w, C, exp_neg_ywx);
  eta_times_grad_times_s = eta * (grad') * s;
  fprintf('loss: %f\n', f_w);

  while true
    test_w = w + alpha * s;
    exp_neg_ywx = exp(-y.* (X * test_w));
    if cost_func(w, C, exp_neg_ywx) <= f_w + alpha * eta_times_grad_times_s
      w = test_w;
      break;
    end
    alpha = alpha ./ 2;
  end
end

function s = conjugate_solve(xi, grad, grad_norm, X, exp_neg_ywx, C)
  s = 0; r = -grad; d = r;
  r_threshold = xi * grad_norm;
  vec_D = exp_neg_ywx ./ ((1 + exp_neg_ywx).^2);

  for i=1:50
    r_norm = norm(r);
    if r_norm <= r_threshold
      break;
    end
    %bottleneck: improve from 45 secs to 4 secs
    % hessian_d = d + (C * X' * (vec_D .* (X * d)));
    hessian_d = d + (C * (vec_D .* (X * d))' * X)';
    r_square = r' * r;
    alpha = r_square ./ (d' * hessian_d);
    s = s + alpha * d;
    r_next = r - alpha * hessian_d;
    beta = (r_next' * r_next) ./ r_square;
    d = r_next + beta * d;
    r = r_next;
  end
  fprintf('CG: %d\n', i);
end
