// worker.js
function getMedian(arr) {
    if (arr.length === 0) return 0;
    const sorted = arr.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0
        ? sorted[mid]
        : Math.round((sorted[mid - 1] + sorted[mid]) / 2);
}

self.onmessage = function(e) {
    if (e.data.type === 'parse_and_analyze') {
        try {
            const jsonStr = e.data.jsonStr;
            self.postMessage({ type: 'progress', message: '正在解析JSON数据...', percent: 5 });
            let data = JSON.parse(jsonStr);
            if (!Array.isArray(data)) data = [data];

            // 汇总统计
            let stats = {
                total: data.length,
                languages: {},
                countries: {},
                timezones: {},
                sources: {},
                placeTypes: {},
                // 不再存数组，只统计数值
                textStats: {
                    count: 0, total: 0, min: Infinity, max: -Infinity
                },
                followerStats: {
                    count: 0, total: 0, min: Infinity, max: -Infinity
                },
                followerArr: [],
                hasGeo: 0,
                hasHashtags: 0,
                hasUrls: 0,
                uniqueUserSet: new Set(),
                timeDistribution: {}
            };
            const batchSize = 10000;
            let processed = 0;

            function analyzeBatch() {
                const end = Math.min(processed + batchSize, data.length);
                for (let i = processed; i < end; i++) {
                    const tweet = data[i];
                    // 语言、国家、时区、来源、地点类型
                    if (tweet.lang) stats.languages[tweet.lang] = (stats.languages[tweet.lang] || 0) + 1;
                    if (tweet.country) stats.countries[tweet.country] = (stats.countries[tweet.country] || 0) + 1;
                    if (tweet.time_zone) stats.timezones[tweet.time_zone] = (stats.timezones[tweet.time_zone] || 0) + 1;
                    if (tweet.source) {
                        let src = tweet.source.replace(/<[^>]*>/g, '');
                        stats.sources[src] = (stats.sources[src] || 0) + 1;
                    }
                    if (tweet.place_type) stats.placeTypes[tweet.place_type] = (stats.placeTypes[tweet.place_type] || 0) + 1;
                    // 文本长度统计
                    if (tweet.text) {
                        let len = tweet.text.length;
                        stats.textStats.count++;
                        stats.textStats.total += len;
                        if (len < stats.textStats.min) stats.textStats.min = len;
                        if (len > stats.textStats.max) stats.textStats.max = len;
                    }
                    // 粉丝统计
                    if (tweet.followers_count) {
                        let n = parseInt(tweet.followers_count);
                        if (!isNaN(n)) {
                            stats.followerStats.count++;
                            stats.followerStats.total += n;
                            if (n < stats.followerStats.min) stats.followerStats.min = n;
                            if (n > stats.followerStats.max) stats.followerStats.max = n;
                            stats.followerArr.push(n);
                        }
                    }
                    // 地理位置
                    if (tweet.coordinates || tweet.geo) stats.hasGeo++;
                    // 标签/链接
                    if (tweet.hashtags && tweet.hashtags !== '[]') stats.hasHashtags++;
                    if (tweet.urls && tweet.urls !== '[]') stats.hasUrls++;
                    // 用户
                    if (tweet.screen_name) stats.uniqueUserSet.add(tweet.screen_name);
                    // 时间分布
                    if (tweet.created_at) {
                        const date = new Date(tweet.created_at);
                        const hour = date.getHours();
                        stats.timeDistribution[hour] = (stats.timeDistribution[hour] || 0) + 1;
                    }
                }
                processed = end;
                const percent = Math.round((processed / data.length) * 95) + 5;
                self.postMessage({ type: 'progress', message: `已处理 ${processed}/${data.length} 条 (${percent}%)`, percent });

                if (processed < data.length) {
                    setTimeout(analyzeBatch, 1);
                } else {
                    // 处理统计收尾
                    stats.textStats.avg = stats.textStats.count ? Math.round(stats.textStats.total / stats.textStats.count) : 0;
                    stats.textStats.min = stats.textStats.count ? stats.textStats.min : 0;
                    stats.textStats.max = stats.textStats.count ? stats.textStats.max : 0;

                    stats.followerStats.avg = stats.followerStats.count ? Math.round(stats.followerStats.total / stats.followerStats.count) : 0;
                    stats.followerStats.min = stats.followerStats.count ? stats.followerStats.min : 0;
                    stats.followerStats.max = stats.followerStats.count ? stats.followerStats.max : 0;
                    stats.followerStats.median = stats.followerArr.length ? getMedian(stats.followerArr) : 0;

                    // 清理Set，防止主线程内存暴涨
                    stats.uniqueUsers = stats.uniqueUserSet.size;
                    delete stats.uniqueUserSet;
                    delete stats.followerArr; // 防止大数组泄漏

                    self.postMessage({ type: 'done', stats });
                }
            }
            setTimeout(analyzeBatch, 1);
        } catch (err) {
            self.postMessage({ type: 'error', message: err.message });
        }
    }
};
