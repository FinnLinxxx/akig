function [mat] = rotY(a)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    mat = [cos(a) 0 -sin(a)
           0 1 0
           sin(a) 0 cos(a)];
end
