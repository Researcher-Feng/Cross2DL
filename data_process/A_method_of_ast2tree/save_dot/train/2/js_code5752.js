function ctor(k, v) {
  if(k && _.isString(k) && k.indexOf('paths.') === 0) return get.apply(null, _.toArray(arguments))
  return v ? set(k, v) : get(k)
}