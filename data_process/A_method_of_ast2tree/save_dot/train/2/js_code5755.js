function load(src) {
  if(!src || !_.isString(src)) return
  let file = _.attempt(require, src)
  if(!file || _.isError(file) || !_.isPlainObject(file)) return
  return _.merge(_SETTINGS, file)
}