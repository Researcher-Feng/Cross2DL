function checkType(type, value, previous, attributeName) {
  var typeDef = parseTypeDef(type);
  var regKey = typeDef.name.toLocaleLowerCase();

  validator = primitives[regKey] || (registry[regKey] && registry[regKey].validator);

  if (!validator) {
    throw TypeException('Unknown type `{{type}}`', null, [ attributeName ], { type: typeDef.name });
  } else if (typeDef.indexes) {
    return arrayValidation(typeDef, 0, value, previous, attributeName, validator);
  }

  return validator(value, previous, attributeName);
}