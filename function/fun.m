function y = fun(x,b,c)
% b: the vector of B_i
% c: the vector of C_i
    y = sum(c./(b+x)) - 1;
end