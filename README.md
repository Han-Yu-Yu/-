#快速幂 
```cpp
ll ksm(ll a,ll n,ll p)
{
    ll s=1;
    while (n)
    {
        if (n&1) s=(s*a)%p;
        a=(a*a)%p;
        n=n>>1;
    }
    return s;
}
```
---
#扩展欧几里得(exgcd)
```cpp
void exgcd(long long a, long long b)
{
	if (b == 0)
	{
		rx = 1;
		d = a;
		ry = 0;
		return;
	}
	exgcd(b, a % b);
	px = rx;
	py = ry;
	rx = py;
	ry = px - py * (a / b);
	return;
}
```
---

#线性求1-n的逆元
```cpp
inv[1] = 1;
for (int i = 2; i <= n; ++i) {
  inv[i] = (long long)(p - p / i) * inv[p % i] % p;
}
```
---

#二维树状数组（区间查询区间修改）
```cpp
#include <bits/stdc++.h>

using namespace std;

typedef long long ll;

const int N = 2048 + 10;
int lowbit(int x)
{
	return (-x) & x;
}
char _;
int n, m;
int tree1[N][N];
int tree2[N][N];
int tree3[N][N];
int tree4[N][N];
void add(int a, int b, int w)
{
	int v1 = w;
	int v2 = w * a;
	int v3 = w * b;
	int v4 = w * a * b;
	for (int i = a; i <= n; i += lowbit(i))
	{
		for (int j = b; j <= m; j += lowbit(j))
		{
			tree1[i][j] += v1;
			tree2[i][j] += v2;
			tree3[i][j] += v3;
			tree4[i][j] += v4;
		}
	}
}
int sum(int a, int b)
{
	int ans = 0;
	for (int i = a; i > 0; i -= lowbit(i))
	{
		for (int j = b; j > 0; j -= lowbit(j))
		{
			ans += tree1[i][j] * (a + 1) * (b + 1);
			ans -= tree2[i][j] * (b + 1);
			ans -= tree3[i][j] * (a + 1);
			ans += tree4[i][j];
		}
	}
	return ans;
}
void add(int a, int b, int c, int d, int w)
{
	add(a, b, w);
	add(a, d + 1, -w);
	add(c + 1, b, -w);
	add(c + 1, d + 1, w);
}

int main()
{
	cin >> _ >> n >> m;
	while (scanf("%c", &_) == 1)
	{
		int a, b, c, d, delta;
		if (_ == 'L')
		{
			cin >> a >> b >> c >> d >> delta;
			add(a, b, c, d, delta);
		}
		else if (_ == 'k')
		{
			cin >> a >> b >> c >> d;
			cout << (sum(c, d) - sum(a - 1, d) - sum(c, b - 1) + sum(a - 1, b - 1)) << endl;
		}
	}
	return 0;
}
```
---

#一维树状数组（区间查询区间修改）
```cpp
#include <bits/stdc++.h>

using namespace std;

typedef long long ll;

const int N = 1e5 + 10;

int n, m;
ll arr[N];

ll tree1[N], tree2[N];

ll lowbit(ll x)
{
	return (-x) & x;
}

void add(ll tree[], ll p, ll x)
{
	for (ll i = p; i <= n; i += lowbit(i))
	{
		tree[i] += x;
	}
	return;
}

ll sum(ll tree[], ll p)
{
	ll ans = 0;
	for (ll i = p; i > 0; i -= lowbit(i))
	{
		ans += tree[i];
	}
	return ans;
}

ll s(ll p)
{
	return sum(tree1, p) * p - sum(tree2, p);
}

ll query(ll l, ll r)
{
	return s(r) - s(l - 1);
}

void build()
{
	for (int i = 1; i <= n; i ++)
	{
		add(tree1, i, arr[i]);
		add(tree1, i + 1, -arr[i]);
		add(tree2, i, (i - 1) * arr[i]);
		add(tree2, i + 1, -i * arr[i]);
	}
}


int main()
{
	cin >> n >> m;
	for (int i = 1; i <= n; i ++)
	{
		cin >> arr[i];
	}
	build();
	while (m--)
	{
		ll cz, x, y, k;
		cin >> cz;
		if (cz == 1)
		{
			cin >> x >> y >> k;
			add(tree1, x, k);
			add(tree1, y + 1, -k);
			add(tree2, x, (x - 1) * k);
			add(tree2, y + 1, -y * k);
		}
		else if (cz == 2)
		{
			cin >> x >> y;
			cout << query(x, y) << endl;
		}
	}
	return 0;
}
```
---

#一维树状数组（区间查询单点修改）
```cpp
#include <bits/stdc++.h>

using namespace std;

typedef long long ll;

const int N = 5e5 + 10;

int n, m;
int tree[N];
int arr[N];

int lowbit(int x)
{
	return x & (-x);
}

void build()
{
	memset(tree, 0, sizeof tree);
	for (int i = 1; i <= n; i ++)
	{
		for (int j = i - lowbit(i) + 1; j <= i; j ++)
		{
			tree[i] += arr[j];
		}
	}
	return;
}

void add(int p, int x)
{
	for (int i = p; i <= n; i += lowbit(i))
	{
		tree[i] += x;
	}
	return;
}

int sum(int x)
{
	int ans = 0;
	for (int i = x; i > 0; i -= lowbit(i))
	{
		ans += tree[i];
	}
	return ans;
}

int query(int l, int r)
{
	return sum(r) - sum(l - 1);
}


int main()
{
	cin >> n >> m;
	for (int i = 1; i <= n; i ++)
	{
		cin >> arr[i];
	}
	build();
	while (m--)
	{
		int cz, x, y;
		cin >> cz >> x >> y;
		if (cz == 1)
		{
			add(x, y);
		}
		else if (cz == 2)
		{
			cout << query(x, y) << endl;
		}
	}
	return 0;
}
```
---


#一维树状数组（单点查询区间修改）
```cpp
#include <bits/stdc++.h>

using namespace std;

typedef long long ll;

const int N = 5e5 + 10;

int n, m;
int tree[N];
int arr[N];

int lowbit(int x)
{
	return x & (-x);
}

void add(int p, int x)
{
	for (int i = p; i <= n; i += lowbit(i))
	{
		tree[i] += x;
	}
	return;
}

void build()
{
	memset(tree, 0, sizeof tree);
	for (int i = 1; i <= n; i ++)
	{
		add(i, arr[i]);
		add(i + 1, -arr[i]);
	}
	return;
}



int sum(int x)
{
	int ans = 0;
	for (int i = x; i > 0; i -= lowbit(i))
	{
		ans += tree[i];
	}
	return ans;
}

int query(int l, int r)
{
	return sum(r) - sum(l - 1);
}


int main()
{
	cin >> n >> m;
	for (int i = 1; i <= n; i ++)
	{
		cin >> arr[i];
	}
	build();
	while (m--)
	{
		int cz, x, y, k;
		cin >> cz;
		if (cz == 1)
		{
			cin >> x >> y >> k;
			add(x, k);
			add(y + 1, -k);
		}
		else if (cz == 2)
		{
			cin >> x;
			cout << sum(x) << endl;
		}
	}
	return 0;
}
```
---

#扩展中国剩余定理（excrt）
```cpp
const int N = 1e5 + 10;
typedef long long ll;
ll x, y, d;
int n;
ll m[N], r[N];
ll gui(ll a, ll b, ll p) // 龟速乘防止爆longlong
{
	a = (a % p + p) % p;
	ll res = 0;
	while (b)
	{
		if (b & 1)
			res = (res + a) % p;
		b /= 2;
		a = (a + a) % p;
	}
	return res;
}
void exgcd(ll a, ll b){if (b == 0){x = 1;y = 0;d = a;return;}exgcd(b, a % b);ll tx = x;ll ty = y;x = ty;y = tx - ty * (a / b);}

ll excrt()
{
	ll tail = 0, llcm = 1, tmp, b, c, x0;
	// ans = lcm * x + tail
	for (int i = 0; i < n; i++)
	{
		// ans = m[i] * y + ri
		// lcm * x + m[i] * y = ri - tail
		// a = lcm
		// b = m[i]
		// c = ri - tail
		b = m[i];
		c = ((r[i] - tail) % b + b) % b;
		exgcd(llcm, b);
		if (c % d)
		{
			return -1;
		}
		// ax + by = gcd(a,b)，特解是，x变量
		// ax + by = c，特解是，x变量 * (c/d)
		// ax + by = c，最小非负特解x0 = (x * (c/d)) % (b/d) 取非负余数
		// 通解 = x0 + (b/d) * n
		x0 = gui(x, c / d, b / d);
	    // 最小非负特解x0
		// ans = lcm * x + tail，带入通解
		// ans = lcm * (x0 + (b/d) * n) + tail
		// ans = lcm * (b/d) * n + lcm * x0 + tail
		// tail' = tail' % lcm'
		tmp = llcm * (b / d);
		tail = (tail + gui(llcm, x0, tmp)) % tmp;
		llcm = tmp;
	}
	return tail;
}
```
---

#中国剩余定理（crt）
```cpp
ll crt()
{
	ll llcm = 1;
	ll ans = 0, ci, ai;
	for (int i = 0; i < n; i ++)
	{
		llcm *= m[i];
	}
	for (int i = 0; i < n; i++)
	{
		ai = llcm / m[i];
		exgcd(ai, m[i]);
		x = (x % m[i] + m[i]) % m[i]; // 确保逆元为正数
		ci = gui(r[i] , gui(ai , x , llcm), llcm);
		ans = (ans + ci) % llcm;
	}
	return ans;
}
```
---

#kn+1法判质数（比试除法快3~4倍左右）
```cpp
bool isPrime(ll n){
    if(n == 2 || n == 3 || n == 5)return 1;
    if(n % 2 == 0 || n % 3 == 0 || n % 5 == 0 || n == 1) return 0;
    ll c = 7, a[8] = {4,2,4,2,4,6,2,6};
    while(c * c <= n) for(auto i : a){if(n % c == 0)return 0; c += i;}
    return 1;
}
```
---

#大步小步算法（BSDS）（解决离散对数问题，分块思想）
```cpp
ll ksm(ll a, ll b, ll mod)
{
	ll s = 1;
	while (b)
	{
		if (b & 1)
			s = s * a % mod;
		a = a * a % mod;
		b >>= 1;
	}
	return s;
}
ll p, b, n;
void solve()
{
	// b ^ x = n (mod p)
	cin >> p >> b >> n;
	if (n == 0)
	{
		if (b % p == 0)
		{
			cout << 1 << endl;
			return;
		}
		cout << "no solution" << endl;
		return;
	}
	if (b == 0)
	{
		cout << "no solution" << endl;
		return;
	}
	if (n == 1)
	{
		cout << 0 << endl;
		return;
	}
	if (b % p == 1 || b % p == 0)
	{
		cout << "no solution" << endl;
		return;
	}
	unordered_map<ll, ll> mp;
	ll m = (ll)sqrt(p) + 1;
	for (int i = 0; i < m; i ++)
	{
		mp[n % p * ksm(b, i, p) % p] = i;
	}
	for (int i = 1; i <= m; i++)
	{
		if (mp.find(ksm(b, i * m, p)) != mp.end())
		{
			cout << i * m - mp[ksm(b, i * m, p)] << endl;
			return;
		}
	}
	cout << "no solution" << endl;
	return;
}
```
---

#扩展大步小步(exBSGS) (a与p不互质)
```cpp
ll exBSGS(ll a, ll p, ll b) // a ^ x = b (mod p)
{
	a %= p;
	b %= p;
	if (b == 1 || p == 1)
		return 0;
	ll d, k = 0, A = 1;
	while (1)
	{
		d = gcd(a, p);
		if (d == 1)
			break;
		if (b % d)
			return -1;
		k++;
		b /= d;
		p /= d;
		A = A * (a / d) % p;
		if (A == b)
			return k;
	} // A*a^(im - j) = b / d (mod p / d)
	ll m = ceil(sqrt(p));
	ll t = b;
	unordered_map<ll, ll> hash;
	hash.reserve(m);
	hash[b] = 0;
	for (int j = 1; j < m; j++)
	{
		t = t * a % p;
		hash[t] = j;
	}
	ll mi = 1;
	for (int i = 1; i <= m; i++)
	{
		mi = mi * a % p;
	}
	t = A;
	for (int i = 1; i <= m; i++)
	{
		t = t * mi % p;
		if (hash.count(t))
		{
			return i * m - hash[t] + k;
		}
	}
	return -1;
}
```
---
#Miller-Rabin素性检验（klogn）
```cpp
std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());

long long fast_mul( long long a, long long b, long long mod ) {
    long long ret = 0;
    a %= mod;
    while( b ){
        if ( b & 1 ) ret = ( ret + a ) % mod;
        b >>= 1;
        a = ( a + a ) % mod;
    }
    return ret;
}


ll ksm(ll a, ll b, ll m)
{
	ll s = 1;
	while (b)
	{
		if (b & 1)
			s = fast_mul(s, a, m);
		a = fast_mul(a, a, m);
		b >>= 1;
	}
	return s;
}

bool Miller_Rabin(ll n)
{
	if (n == 2  || n == 3 )
	{
		return true;
	}
	if (n == 1 || n % 2 == 0)
	{
		return false;
	}
	long long d = n - 1LL;
	int s = 0;
	while (d % 2 == 0)
		s++, d >>= 1;
	for (int i = 0; i < 10; i ++)
	{
		long long a = rand() % (n - 3) + 2;
		long long x = ksm( a, d, n );
        long long y = 0;
        for(int j = 0; j < s; j++ ) {
            y = fast_mul( x, x, n );
            if ( y == 1 && x != 1 && x != n - 1 ) return false;
            x = y;
        }
        if ( y != 1 ) return false;

	}
	return true;
}
```
---
#Miller-Rabin素性检验简化版（4logn）（int内）
```cpp
ll ksm(ll a, ll b, ll m)
{
	ll s = 1;
	while (b > 0)
	{
		if (b & 1) s = s * a % m;
		a = a * a % m;
		b >>= 1;
	}
	return s;
}

bool Miller_Rabin(ll x)
{
	if (x % 2 == 0 || x % 3 == 0 || x % 5 == 0 || x % 7 == 0)
	{
		return false;
	}
	ll temp[5] = {2, 3, 5, 7};
	for (int i = 0; i < 4; i ++)
	{
		if (ksm(temp[i], x - 1, x) != 1)
		{
			return false;
		}
	}
	return true;
}
```
---
#int_128
```cpp
inline void input(__int128 &s)
{
    s = 0;
    char c = ' ';
    while (c > '9' || c < '0')
        c = getchar();
    while (c >= '0' && c <= '9')
    {
        s = s * 10 + c - '0';
        c = getchar();
    }
}

inline void print(__int128 x)
{
    if (x < 0)
    {
        putchar('-');
        x = -x;
    }
    if (x > 9)
        print(x / 10);
    putchar(x % 10 + '0');
}
```
---
#区间改查线段树
```cpp
int n, m;
ll tree[N << 2];
ll add[N << 2];
ll arr[N];

void up(int x)
{
	tree[x] = tree[x << 1] + tree[x << 1 | 1];
	return;
}

void lazy(int i, ll v, int n)
{
	tree[i] += v * n;
	add[i] += v;
}

void down(int i, int ln, int rn)
{
	if (add[i])
	{
		lazy(i << 1, add[i], ln);
		lazy(i << 1 | 1, add[i], rn);
		add[i] = 0;
	}
	return;
}

void build(int l, int r, int i)
{
	if (l == r)
	{
		tree[i] = arr[l];
	}
	else
	{
		int mid = l + r >> 1;
		build(l, mid, i << 1);
		build(mid + 1, r, i << 1 | 1);
		up(i);
	}
	add[i] = 0;
}

void ad(int jobl, int jobr, ll jobv, int l, int r, int i)
{
	if (l >= jobl && r <= jobr)
	{
		lazy(i, jobv, r - l + 1);
	}
	else
	{
		int mid = l + r >> 1;
		down(i, mid - l + 1, r - mid);
		if (jobl <= mid)
		{
			ad(jobl, jobr, jobv, l, mid, i << 1);
		}
		if (jobr > mid)
		{
			ad(jobl, jobr, jobv, mid + 1, r, i << 1 | 1);
		}
		up(i);
	}
}

ll query(int jobl, int jobr, int l, int r, int i)
{
	if (l >= jobl && r <= jobr)
	{
		return tree[i];
	}
	int mid = l + r >> 1;
	down(i, mid - l + 1, r - mid);
	ll ans = 0;
	if (jobl <= mid)
	{
		ans += query(jobl, jobr, l, mid, i << 1);
	}
	if (jobr > mid)
	{
		ans += query(jobl, jobr, mid + 1, r, i << 1 | 1);
	}
	return ans;
}
```
---
#随机数
```cpp
ull p[350010];
void init_hash() {
    mt19937_64 rng(time(0));
    for (int i = 0; i < 350010; ++i) {
        p[i] = rng();  // 使用随机数生成器
    }
}

梅森素数哈希，通过异或可以得到唯一的另一个数：
std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());
```
---

#倍增求lca（nlogn）
```cpp
const int N = 5e5 + 10;
int n, m, s;
vector<int> e[N];
int dep[N];
int st[20][N];
inline void dfs1(int x)
{
	for (int i = 0; i < e[x].size(); i++)
	{
		if (e[x][i] == st[0][x])
			continue;
		st[0][e[x][i]] = x;
		dep[e[x][i]] = dep[x] + 1;
		dfs1(e[x][i]);
	}
	return;
}

inline void dfs2(int x)
{
	for (int j = 1; j <= 19; j++)
	{
		st[j][x] = st[j - 1][st[j - 1][x]];
		if (dep[st[j][x]] == 0)
			break;
	}
	for (int i = 0; i < e[x].size(); i++)
	{
		if (e[x][i] == st[0][x])
			continue;
		dfs2(e[x][i]);
	}
}

inline void init()
{
	dep[s] = 1;
	dfs1(s);
	dfs2(s);
}

inline int lca(int a, int b)
{
	if (dep[a] < dep[b])
	{
		while (dep[a] < dep[b])
		{
			int p = 19;
			while (p >= 0 && dep[a] > dep[st[p][b]])
			{
				p--;
			}
			if (p == -1)
				break;
			// cout << "p: " << p << endl;
			b = st[p][b];
		}
	}
	else if (dep[a] > dep[b])
	{
		while (dep[a] > dep[b])
		{
			int p = 19;
			while (p >= 0 && dep[st[p][a]] < dep[b])
			{
				p--;
			}
			if (p == -1)
				break;
			a = st[p][a];
		}
	}
	if (a == b)
	{
		return a;
	}
	while (st[0][a] != st[0][b])
	{
		for (int j = 19; j >= 0; j--)
		{
			if (st[j][a] != st[j][b])
			{
				a = st[j][a];
				b = st[j][b];
			}
		}
	}
	return st[0][a];
}
```
---
#tarjan求lca  O(n + q)
```cpp
int n, m, s;
struct question
{
	int to;
	int xh;
};
vector<int> e[N];
vector<question> q[N];
int a[N];
int res[N];
int flag[N];

int find(int x)
{
	if (a[x] == x)
	{
		return x;
	}
	return a[x] = find(a[x]);
}

void dfs(int st, int fa)
{
	for (int i = 0; i < e[st].size(); i++)
	{
		if (e[st][i] == fa)
			continue;
		flag[e[st][i]] = 1;
		for (int j = 0; j < q[e[st][i]].size(); j++)
		{
			if (flag[q[e[st][i]][j].to])
			{
				//cout << e[st][i] << " to: " << q[e[st][i]][j].to << endl;
				res[q[e[st][i]][j].xh] = find(q[e[st][i]][j].to);
			}
		}
		dfs(e[st][i], st);
		a[find(e[st][i])] = st;
	}
}
```
---
#树链剖分求lca（单次O（logn））
```cpp
vector<int> e[N];
int fa[N];
int son[N];
int sz[N];
int top[N];
int dep[N];
void dfs1(int x, int f)
{
	dep[x] = dep[f] + 1;
	sz[x] = 1;
	fa[x] = f;
	for (int i : e[x])
	{
		if (i == f)
			continue;
		dfs1(i, x);
		sz[x] += sz[i];
		if (sz[son[x]] < sz[i])
		{
			son[x] = i;
		}
	}
}

void dfs2(int x, int f)
{
	top[x] = top[f];
	if (!son[x])
	{
		return;
	}
	dfs2(son[x], x);
	for (int i : e[x])
	{
		if (i == son[x] || i == fa[x])
			continue;
		dfs2(i, i);
	}
}

int lca(int a, int b)
{
	while (top[a] != top[b])
	{
		if (dep[top[a]] < dep[top[b]])
			swap(a, b);
		// cout << "a: " << a << " b: " << b << endl;
		if (top[a] != top[b])
			a = fa[top[a]];
	}
	return dep[a] < dep[b] ? a : b;
}

```
---
