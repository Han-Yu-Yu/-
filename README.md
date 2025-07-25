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
