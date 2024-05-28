function set(k, v) {
  let curr = get(k)
  if(curr && _.isPlainObject(curr) && _.isPlainObject(v)) v = _.mcopy(curr, v)
  if(k) _.set(_SETTINGS, k, v)
  if(!k) _SETTINGS = v
  return get(k)
}