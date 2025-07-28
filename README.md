#快速幂 
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
-

#线性求1-n的逆元
inv[1] = 1;
for (int i = 2; i <= n; ++i) {
  inv[i] = (long long)(p - p / i) * inv[p % i] % p;
}
-

#二维树状数组（区间查询区间修改）
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
-

#一维树状数组（区间查询区间修改）
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
-

#一维树状数组（区间查询单点修改）
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
-

#一维树状数组（单点查询区间修改）
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
-
