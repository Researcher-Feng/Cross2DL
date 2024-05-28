function load(store, key, url, options) {
  const { cacheControl, rejectOnError, retry, timeout } = options;

  options.id = key;

  store.debug('load %s from %s', key, url);

  return agent
    .get(url, options)
    .timeout(timeout)
    .retry(retry)
    .then(res => {
      // Abort if already destroyed
      if (store.destroyed) {
        throw Error('store destroyed');
      }

      store.debug('loaded "%s" in %dms', key, res.duration);

      // Guard against empty data
      if (res.body) {
        // Parse cache-control headers
        if (res.headers && 'expires' in res.headers) {
          res.body[store.EXPIRY_KEY] = generateExpiry(res.headers, cacheControl);
        }

        // Enable handling by not calling inner set()
        store.set(key, res.body, options);
      }

      return res;
    })
    .catch(err => {
      // Abort if already destroyed
      if (store.destroyed) {
        throw err;
      }

      store.debug('unable to load "%s" from %s', key, url);

      if (rejectOnError) {
        store.set(key, undefined, options);
      }

      throw err;
    });
}