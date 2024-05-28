function decodeResult(contentType, dataType, result) {
  if(dataType=='json'){
    try {
      result = JSON.parse(result);
      return result;
    } catch (e) {}
  }

  if (contentType && contentType.indexOf('application/json')==0) {
    try {
      result = JSON.parse(result);
    } catch (e) {}
  }
  //todo: xml result
  return result;
}