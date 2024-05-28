function scheduleJob(trigger, jobFunc, jobData) {
    const job = Job.createJob(trigger, jobFunc, jobData);
    const excuteTime = job.excuteTime();
    const id = job.id;

    map[id] = job;
    const element = {
        id: id,
        time: excuteTime
    };

    const curJob = queue.peek();
    if (!curJob || excuteTime < curJob.time) {
        queue.offer(element);
        setTimer(job);

        return job.id;
    }

    queue.offer(element);
    return job.id;
}