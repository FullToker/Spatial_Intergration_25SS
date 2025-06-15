// main.js
let currentJsonStr = '';

window.onload = function() {
    setupFileUpload();
};

function setupFileUpload() {
    const fileUploadZone = document.getElementById('fileUploadZone');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const uploadProgress = document.getElementById('uploadProgress');
    const progressText = document.getElementById('progressText');
    const progressPercent = document.getElementById('progressPercent');
    const progressFill = document.getElementById('progressFill');
    const processedSize = document.getElementById('processedSize');
    const totalSize = document.getElementById('totalSize');
    const remainingTime = document.getElementById('remainingTime');

    fileUploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => handleFileSelect(e.target.files[0]));
    fileUploadZone.addEventListener('dragover', (e) => { e.preventDefault(); fileUploadZone.classList.add('dragover'); });
    fileUploadZone.addEventListener('dragleave', (e) => { e.preventDefault(); fileUploadZone.classList.remove('dragover'); });
    fileUploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        fileUploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleFileSelect(e.dataTransfer.files[0]);
    });

    function handleFileSelect(file) {
        if (!file) return;
        if (!file.name.toLowerCase().endsWith('.json')) {
            showError('请选择JSON格式的文件');
            return;
        }
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        fileInfo.style.display = 'block';

        uploadProgress.style.display = 'block';
        progressText.textContent = '正在读取文件...';
        progressPercent.textContent = '0%';
        progressFill.style.width = '0%';
        totalSize.textContent = formatFileSize(file.size);
        processedSize.textContent = '0 MB';
        remainingTime.textContent = '计算中...';

        readFileWithProgress(file);
    }

    function readFileWithProgress(file) {
        const chunkSize = 2 * 1024 * 1024; // 2MB
        const totalChunks = Math.ceil(file.size / chunkSize);
        let currentChunk = 0;
        let result = '';
        const startTime = Date.now();

        function readNextChunk() {
            const start = currentChunk * chunkSize;
            const end = Math.min(start + chunkSize, file.size);
            const chunk = file.slice(start, end);

            const reader = new FileReader();
            reader.onload = (e) => {
                result += e.target.result;
                currentChunk++;

                // 更新进度
                const progress = (currentChunk / totalChunks) * 100;
                const processedBytes = Math.min(currentChunk * chunkSize, file.size);

                progressFill.style.width = progress + '%';
                progressPercent.textContent = Math.round(progress) + '%';
                processedSize.textContent = formatFileSize(processedBytes);

                // 剩余时间估算
                if (currentChunk > 1) {
                    const elapsed = Date.now() - startTime;
                    const rate = processedBytes / elapsed;
                    const remaining = (file.size - processedBytes) / rate;
                    remainingTime.textContent = formatTime(remaining);
                }

                if (currentChunk < totalChunks) {
                    setTimeout(readNextChunk, 5);
                } else {
                    progressText.textContent = '文件读取完成！';
                    currentJsonStr = result;
                    setTimeout(() => {
                        uploadProgress.style.display = 'none';
                        showSuccess(`文件 "${file.name}" 已成功加载！(${formatFileSize(file.size)})`);
                        startAnalyze(currentJsonStr, file.size);
                    }, 400);
                }
            };
            reader.onerror = () => {
                showError('文件读取失败');
                uploadProgress.style.display = 'none';
            };
            reader.readAsText(chunk);
        }
        readNextChunk();
    }
}

function startAnalyze(jsonStr, fileSize) {
    // 超过5MB直接后台分析，不展示JSON原文
    if (fileSize < 5 * 1024 * 1024) {
        // 小文件，可以用alert(JSON长度)或者放到文本区等
    } else {
        showWarning('文件较大，已直接载入内存分析。');
    }

    showAnalysisProgress();

    // 启动 Web Worker
    const worker = new Worker('worker.js');
    worker.postMessage({ type: 'parse_and_analyze', jsonStr });
    worker.onmessage = function(e) {
        if (e.data.type === 'progress') {
            showAnalysisStatus(e.data.message);
            updateAnalysisProgress(e.data.percent);
        }
        else if (e.data.type === 'done') {
            hideAnalysisProgress();
            displayResults(e.data.stats);
            worker.terminate();
        }
        else if (e.data.type === 'error') {
            hideAnalysisProgress();
            showError('JSON解析错误：' + e.data.message);
            worker.terminate();
        }
    };
}

function showAnalysisProgress() {
    document.getElementById('results').innerHTML = `
        <div id="analysisProgress" style="margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 15px; text-align: center;">
            <div style="font-size: 1.2em; font-weight: 600; color: #495057; margin-bottom: 15px;">
                🔄 正在分析数据...
            </div>
            <div class="progress-bar" style="height: 12px; margin-bottom: 10px;">
                <div id="analysisProgressFill" class="progress-fill" style="width: 0%;"></div>
            </div>
            <div id="analysisStatus" style="color: #666; font-size: 0.9em;">
                准备开始...
            </div>
        </div>`;
    document.getElementById('results').style.display = 'block';
}
function showAnalysisStatus(message) {
    const statusElement = document.getElementById('analysisStatus');
    if (statusElement) statusElement.textContent = message;
}
function updateAnalysisProgress(percent) {
    const progressFill = document.getElementById('analysisProgressFill');
    if (progressFill) progressFill.style.width = percent + '%';
}
function hideAnalysisProgress() {
    const analysisProgress = document.getElementById('analysisProgress');
    if (analysisProgress) analysisProgress.remove();
}

function showError(msg) {
    const errorDiv = document.getElementById('error');
    errorDiv.innerHTML = `<div class="error">❌ ${msg}</div>`;
    errorDiv.style.display = 'block';
}
function showSuccess(msg) {
    const errorDiv = document.getElementById('error');
    errorDiv.innerHTML = `<div style="background: #f0f9ff; border: 1px solid #0ea5e9; color: #0c4a6e; padding: 15px; border-radius: 10px; margin: 10px 0;">✅ ${msg}</div>`;
    errorDiv.style.display = 'block';
    setTimeout(() => { errorDiv.style.display = 'none'; }, 3000);
}
function showWarning(msg) {
    const errorDiv = document.getElementById('error');
    errorDiv.innerHTML = `<div style="background: #fffbeb; border: 1px solid #f59e0b; color: #92400e; padding: 15px; border-radius: 10px; margin: 10px 0;">⚠️ ${msg}</div>`;
    errorDiv.style.display = 'block';
    setTimeout(() => { errorDiv.style.display = 'none'; }, 5000);
}
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024, sizes = ['Bytes','KB','MB','GB'], i = Math.floor(Math.log(bytes)/Math.log(k));
    return parseFloat((bytes/Math.pow(k,i)).toFixed(2)) + ' ' + sizes[i];
}
function formatTime(ms) {
    if (ms < 1000) return '< 1秒';
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    if (hours > 0) return `${hours}小时${minutes % 60}分钟`;
    if (minutes > 0) return `${minutes}分钟${seconds % 60}秒`;
    return `${seconds}秒`;
}

function displayResults(stats) {
    const resultsDiv = document.getElementById('results');
    let html = `
        <div class="summary">
            <h2>📋 数据概览</h2>
            <div class="summary-grid">
                <div class="summary-item"><span class="summary-number">${stats.total}</span><span class="summary-label">总推文数</span></div>
                <div class="summary-item"><span class="summary-number">${stats.uniqueUsers}</span><span class="summary-label">独立用户</span></div>
                <div class="summary-item"><span class="summary-number">${Object.keys(stats.languages).length}</span><span class="summary-label">语言类型</span></div>
                <div class="summary-item"><span class="summary-number">${Object.keys(stats.countries).length}</span><span class="summary-label">国家数量</span></div>
                <div class="summary-item"><span class="summary-number">${stats.textStats.avg}</span><span class="summary-label">平均字符数</span></div>
                <div class="summary-item"><span class="summary-number">${Math.round((stats.hasGeo / stats.total) * 100)}%</span><span class="summary-label">有地理位置</span></div>
            </div>
        </div>
        <div class="stats-grid">`;
    html += createStatCard('🌐 语言分布', stats.languages, stats.total);
    html += createStatCard('🌍 国家分布', stats.countries, stats.total);
    html += createStatCard('🕐 时区分布', stats.timezones, stats.total);
    html += createStatCard('📱 发布来源', stats.sources, stats.total);
    if (Object.keys(stats.placeTypes).length > 0) html += createStatCard('📍 地点类型', stats.placeTypes, stats.total);

    // 文本统计
    html += createStatCard('📝 文本统计', {
        '平均长度': stats.textStats.avg,
        '最短': stats.textStats.min,
        '最长': stats.textStats.max,
        '总字符数': stats.textStats.total
    });

    // 粉丝统计
    html += createStatCard('👥 粉丝统计', {
        '平均粉丝数': stats.followerStats.avg,
        '最少粉丝': stats.followerStats.min,
        '最多粉丝': stats.followerStats.max,
        '中位数': stats.followerStats.median
    });

    const contentStats = {
        '包含地理位置': `${stats.hasGeo} (${Math.round((stats.hasGeo / stats.total) * 100)}%)`,
        '包含标签': `${stats.hasHashtags} (${Math.round((stats.hasHashtags / stats.total) * 100)}%)`,
        '包含链接': `${stats.hasUrls} (${Math.round((stats.hasUrls / stats.total) * 100)}%)`
    };
    html += createStatCard('🏷️ 内容特征', contentStats);

    if (Object.keys(stats.timeDistribution).length > 0) {
        html += createStatCard('⏰ 发布时间分布（24小时制）', stats.timeDistribution, stats.total);
    }
    html += '</div>';
    resultsDiv.innerHTML = html;
    resultsDiv.style.display = 'block';
}
function createStatCard(title, data, total = null) {
    let html = `<div class="stat-card"><div class="stat-title">${title}</div>`;
    if (typeof data === 'object') {
        const sortedEntries = Object.entries(data)
            .sort(([,a],[,b]) => b-a)
            .slice(0, 10);
        sortedEntries.forEach(([key, value]) => {
            const percentage = total ? Math.round((value / total) * 100) : null;
            html += `<div class="stat-item"><span class="stat-label">${key}</span><span class="stat-value">${value}${percentage ? ` (${percentage}%)` : ''}</span></div>`;
            if (total && percentage) html += `<div class="progress-bar"><div class="progress-fill" style="width: ${percentage}%"></div></div>`;
        });
        if (Object.keys(data).length > 10)
            html += `<div class="stat-item"><span class="stat-label">... 还有 ${Object.keys(data).length - 10} 项</span></div>`;
    }
    html += '</div>';
    return html;
}

function getMedian(arr) {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : Math.round((sorted[mid - 1] + sorted[mid]) / 2);
}
