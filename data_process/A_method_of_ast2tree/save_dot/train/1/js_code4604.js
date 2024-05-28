function mergeCacheControl(cacheControl, defaultCacheControl) {
  if (cacheControl == null) {
    return Object.assign({}, defaultCacheControl);
  }

  return {
    maxAge: 'maxAge' in cacheControl ? cacheControl.maxAge : defaultCacheControl.maxAge,
    staleIfError: 'staleIfError' in cacheControl ? cacheControl.staleIfError : defaultCacheControl.staleIfError
  };
}