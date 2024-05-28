function generateResponseHeaders(expiry = {}, defaultCacheControl, isError) {
  const now = Date.now();
  let maxAge;

  if (isError) {
    maxAge =
      expiry && expiry.expiresIfError > now && expiry.expiresIfError - now < defaultCacheControl.maxAge
        ? Math.ceil((expiry.expiresIfError - now) / 1000)
        : defaultCacheControl.maxAge / 1000;
  } else {
    // Round up to nearest second
    maxAge =
      expiry && expiry.expires > now ? Math.ceil((expiry.expires - now) / 1000) : defaultCacheControl.maxAge / 1000;
  }

  return {
    // TODO: add stale-if-error
    'cache-control': `public, max-age=${maxAge}`,
    expires: new Date(now + maxAge * 1000).toUTCString()
  };
}