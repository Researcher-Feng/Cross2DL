function defaultMapFn(data) {
  return Object.keys(data).slice(0, this.headers.length).map(function(key) {
    return data[key]
  })
}