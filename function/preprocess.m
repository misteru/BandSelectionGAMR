function [data_mat, label_mat] = preprocess(data, label, w, h)

[m,n,b] = size(data);

if m > h
    m1 = randi([1, m - h]);
    m2 = m1 + h - 1;
else
    m1 = 1;
    m2 = h;
end
if n > w
    n1 = randi([1, n - w]);
    n2 = n1 + w - 1;
else
    n1 = 1;
    n2 = w;
end

data_mat = data(m1:m2, n1:n2, :);
label_mat = label(m1:m2, n1:n2);
% reshape to size(n_sample, n_feature)
data_mat = reshape(data_mat, [], b);
label_mat = reshape(label_mat, [], 1);
% permute to size(n_feature, n_sample)
data_mat = permute(data_mat, [2,1]);
% normalize (calculate each column of data separately)
data_mat = normalize(data_mat, "norm");
% permute to size(n_sample, n_feature)
data_mat = permute(data_mat, [2,1]);
end


