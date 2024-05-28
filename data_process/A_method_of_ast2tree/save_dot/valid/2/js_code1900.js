function dataType (value) {
  if (!value) return null
  if (value['anyOf'] || value['allOf'] || value['oneOf']) {
    return ''
  }
  if (!value.type) {
    return 'object'
  }
  if (value.type === 'array') {
    return dataType(value.items || {}) + '[]'
  }
  return value.type
}