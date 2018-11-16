%module tandems

%{
#include "tandems.hpp"
%}

%include "std_string.i"
%include "std_vector.i"
%include "std_pair.i"
std::string replace_tandems(std::string s, std::string split_sep=" ", std::string join_sep=" ");