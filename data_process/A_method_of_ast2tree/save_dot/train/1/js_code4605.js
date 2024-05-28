function generateExpiry(headers = {}, defaultCacheControl) {
  const cacheControl = mergeCacheControl(parseCacheControl(headers['cache-control']), defaultCacheControl);

  const now = Date.now();
  let expires = now;

  if (headers.expires) {
    expires = typeof headers.expires === 'string' ? Number(new Date(headers.expires)) : headers.expires;
  }
  if (now >= expires) {
    expires = now + cacheControl.maxAge;
  }

  return {
    expires,
    expiresIfError: expires + cacheControl.staleIfError
  };
}