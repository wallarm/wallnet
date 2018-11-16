#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

using namespace std;

vector<string> sub_vec(vector<string> s, int begin, int end)
{
	vector<string> new_vec;

	for (size_t i = begin; i < end; i++) {
		new_vec.push_back(s[i]);
	}
	return new_vec;
}

vector<string> concat(vector<string> v, vector<string> u)
{
	vector<string> new_vec = v;
	new_vec.insert(new_vec.end(), u.begin(), u.end());
	return new_vec;
}

vector<string> concat(vector<string> v, string u)
{
	vector<string> new_vec = v;
	new_vec.push_back(u);
	return new_vec;
}

string join(const vector<string> &s, const string &sep)
{
	string joined_string = "";

	for (size_t i = 0; i < (int)s.size() - 1; i++) {
		joined_string += s[i] + sep;
	}
	joined_string += s[(int)s.size() - 1];
	return joined_string;
}

string join(const vector<pair<int, int>> &s, const string &sep)
{
	string joined_string = "";

	for (size_t i = 0; i < (int)s.size() - 1; i++) {
		joined_string += to_string(s[i].first) + ", " + to_string(s[i].second) + sep;
	}
	joined_string += to_string(s[s.size() - 1].first) + ", " + to_string(s[s.size() - 1].second);
	return joined_string;
}

vector<int> interval_sizes(const vector<pair<int, int>> &s)
{
	vector<int> sizes;

	for (int i = 0; i < s.size(); i++) {
		sizes.push_back(s[i].second - s[i].first);
	}
	return sizes;
}


vector<int> z_function(vector<string> s)
{
	int n = (int)s.size();
	vector<int> z(n);
	for (int i = 1, l = 0, r = 0; i<n; ++i) {
		if (i <= r)
			z[i] = min(r - i + 1, z[i - l]);
		while (i + z[i] < n && s[z[i]] == s[i + z[i]])
			++z[i];
		if (i + z[i] - 1 > r)
			l = i, r = i + z[i] - 1;
	}
	return z;
}


void save_tandem(const vector<string> & s, int shift, bool left, int cntr, int l, int l1, int l2, vector<pair<int, int>> & intervals)
{
	int pos;
	if (left)
		pos = cntr - l1;
	else
		pos = cntr - l1 - l2 - l1 + 1;
	pair<int, int> interval;
	interval.first = shift + pos;
	interval.second = l;
	intervals.push_back(interval);
}


void save_tandems(const vector<string> & s, int shift, bool left, int cntr, int l, int k1, int k2, vector<pair<int, int>> & intervals)
{
	for (int l1 = 1; l1 <= l; ++l1) {
		if (left && l1 == l)  break;
		if (l1 <= k1 && l - l1 <= k2)
			save_tandem(s, shift, left, cntr, l, l1, l - l1, intervals);
	}
}

inline int get_z(const vector<int> & z, int i)
{
	return 0 <= i && i<(int)z.size() ? z[i] : 0;
}


void find_tandems(vector<string> s, vector<pair<int, int>> & intervals, int shift = 0)
{
	int n = (int)s.size();
	if (n == 1) {
		return;
	}

	int nu = n / 2, nv = n - nu;
	vector<string> u = sub_vec(s, 0, nu),
		v = sub_vec(s, nu, (int)s.size());
	vector<string> ru = u, rv = v;
	reverse(ru.begin(), ru.end());
	reverse(rv.begin(), rv.end());

	find_tandems(u, intervals, shift);
	find_tandems(v, intervals, shift + nu);

	vector<int> z1 = z_function(ru),
		z2 = z_function(concat(concat(v, "#"), u)),
		z3 = z_function(concat(concat(ru, "#"), rv)),
		z4 = z_function(v);


	for (int cntr = 0; cntr<n; ++cntr) {
		int l, k1, k2;
		if (cntr < nu) {
			l = nu - cntr;
			k1 = get_z(z1, nu - cntr);
			k2 = get_z(z2, nv + 1 + cntr);
		}
		else {
			l = cntr - nu + 1;
			k1 = get_z(z3, nu + 1 + nv - 1 - (cntr - nu));
			k2 = get_z(z4, (cntr - nu) + 1);
		}
		if (k1 + k2 >= l)
			save_tandems(s, shift, cntr<nu, cntr, l, k1, k2, intervals);
	}
}

void find_and_replace(vector<string> &a)
{
	vector<pair<int, int>> intervals; // first - start position of tandem; second - length of tandem.
	find_tandems(a, intervals);
	vector<int> sizes = interval_sizes(intervals);
	if (sizes.size() != 0) {
		int num_interval = distance(sizes.begin(), max_element(sizes.begin(), sizes.end()));
		int erase_beg, erase_end;
		erase_beg = intervals[num_interval].first;
		erase_end = intervals[num_interval].first + intervals[num_interval].second;
		a.erase(a.begin() + erase_beg, a.begin() + erase_end);
		//cout << ">> " << join(a, "|") << endl;
	}
}

vector<string> replace_tandems(vector<string> a) {
	vector<string> _a;
	while (_a != a) {
		_a = a;
		find_and_replace(a);
	}
	return a;
}

vector<string> split(const string& text, const string& delims)
{
	vector<string> tokens;
	size_t start = text.find_first_not_of(delims), end = 0;

	while ((end = text.find_first_of(delims, start)) != std::string::npos)
	{
		tokens.push_back(text.substr(start, end - start));
		start = text.find_first_not_of(delims, end);
	}
	if (start != std::string::npos)
		tokens.push_back(text.substr(start));

	return tokens;
}

string replace_tandems(string s, string split_sep=" ", string join_sep=" ") {
	if (s == "" or s == split_sep) {
        return join_sep;
    } 
	else {
		return join(replace_tandems(split(s, split_sep)), join_sep);
	}
}

// int main(int argc, char const *argv[])
// {

// 	string a = " SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    NUM   SINGLE_CHAR    SINGLE_CHAR   po sadw qt  SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    NUM   SINGLE_CHAR    SINGLE_CHAR   po sadw qt  SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    NUM   SINGLE_CHAR    SINGLE_CHAR   po sadw qt  SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    NUM   SINGLE_CHAR    SINGLE_CHAR   po sadw qt  SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    NUM   SINGLE_CHAR    SINGLE_CHAR   po sadw qt  SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    NUM   SINGLE_CHAR    SINGLE_CHAR   po sadw qt  SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    NUM   SINGLE_CHAR    SINGLE_CHAR   po sadw qt  SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    NUM   SINGLE_CHAR    SINGLE_CHAR   po sadw qt  SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    NUM   SINGLE_CHAR    SINGLE_CHAR   po sadw qt  SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    SINGLE_CHAR    NUM   SINGLE_CHAR    SINGLE_CHAR   po sadw  SINGLE_CHAR  ";
// 	cout << a << endl;
// 	string _a = replace_tandems(a);
// 	cout << endl << _a << endl;

// 	return 0;
// }