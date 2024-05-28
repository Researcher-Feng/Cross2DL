function hasExpired(expiry, isError) {
  if (!expiry) {
    return true;
  }

  // Round up to nearest second
  return Math.ceil(Date.now() / 1000) * 1000 > (isError ? expiry.expiresIfError : expiry.expires);
}