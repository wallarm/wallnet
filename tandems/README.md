# Replace Tandems
It's a programm with Python wrapper for removing all tandem repeats in string. 
Tandem repeat it's a two occurrences of any substring in a string.

The search is carried out using the [Main-Lorentz algorithm](http://e-maxx.ru/algo/string_tandems)

# Build
```
swig -c++ -python tandems.i && g++ -std=c++11 -fpic -c tandems.hpp tandems_wrap.cxx -I/usr/include/python3.6/ && gcc -shared tandems_wrap.o -o _tandems.so -lstdc++ -std=c++11
```

# Usage
if 
```
import _tandems
new_string = _tandems.replace_tandems(string, split_sep, join_sep)
```

Removing trash
```
rm tandems.py  _tandems.so tandems_wrap.* tandems.hpp.gch

```