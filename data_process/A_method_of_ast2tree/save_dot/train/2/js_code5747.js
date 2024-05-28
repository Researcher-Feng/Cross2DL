function undefineType(type) {
  var validator;
  var typeDef = parseTypeDef(type);
  var regKey = typeDef.name.toLocaleLowerCase();

  if (primitives[regKey]) {
    throw TypeException('Cannot undefine primitive type `{{type}}`', null, null, { type: typeDef.name });
  }

  validator = registry[regKey] && registry[regKey].validator;

  delete registry[regKey];

  return validator || false;
}