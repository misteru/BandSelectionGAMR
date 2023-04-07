function [data_mat, label_mat] = preprocess2(data, label, w, h)

[~,~,b] = size(data);


data_mat = data;
label_mat = label;
% reshape to size(n_sample, n_feature)
data_mat = reshape(data_mat, [], b);
label_mat = reshape(label_mat, [], 1);
m=max(label_mat);n=min(label_mat);
ind1 =[];
for i=n:m
    index=find(label_mat==i);
    ind2=randperm(length(index));
    ind1=[ind1; index(ind2(1:w))];
end
data_mat=data_mat(ind1,:);
label_mat=label_mat(ind1);

% permute to size(n_feature, n_sample)
data_mat = permute(data_mat, [2,1]);
% normalize (calculate each column of data separately)
data_mat = normalize(data_mat, "norm");
% permute to size(n_sample, n_feature)
data_mat = permute(data_mat, [2,1]);
end


