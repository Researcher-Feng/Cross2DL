function parseCacheControl(cacheControlString) {
  let maxAge = 0;
  let staleIfError = 0;

  if (cacheControlString && typeof cacheControlString === 'string') {
    let match;

    while ((match = RE_CACHE_CONTROL.exec(cacheControlString))) {
      if (match[1]) {
        maxAge = parseInt(match[1], 10) * 1000;
      } else if (match[2]) {
        staleIfError = parseInt(match[2], 10) * 1000;
      }
    }
  }

  return {
    maxAge,
    staleIfError
  };
}