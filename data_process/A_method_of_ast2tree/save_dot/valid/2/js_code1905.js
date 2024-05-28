function forceUpdateSuiteData(suites, test) {
    const id = getSuiteId(test);
    suites[id] = cloneDeep(suites[id]);
}