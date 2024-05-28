function get(k) {
  if(!k) return _SETTINGS
  let v = _.get(_SETTINGS, k)
  if(!v) return
  if(_.isString(k) && k.indexOf('paths.') !== 0) return v
  let args = _.drop(_.toArray(arguments))
  let argsLength = args.unshift(v)
  return path.join.apply(path, args)
}