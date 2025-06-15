// 全局变量
let tweetData = [];
let analysisWorker = null;
let isAnalyzing = false;
let fileReader = null;

// 创建Web Worker
function createAnalysisWorker() {
    if (analysisWorker) {
        analysisWorker.terminate();
    }
    
    const workerCode = `
        self.onmessage = function(e) {
            const { data, batchSize } = e.data;
            const totalTweets = data.length;
            let processedCount = 0;
            
            const stats = {
                total: totalTweets,
                languages: {},
                countries: {},
                timezones: {},
                sources: {},
                placeTypes: {},
                textLengths: [],
                followerCounts: [],
                coordinates: [],
                hasGeo: 0,
                hasHashtags: 0,
                hasUrls: 0,
                uniqueUsers: new Set(),
                timeDistribution: {}
            };
            
            function processBatch() {
                const endIndex = Math.min(processedCount + batchSize, totalTweets);
                const batch = data.slice(processedCount, endIndex);
                
                batch.forEach(tweet => {
                    // 语言统计
                    if (tweet.lang) {
                        stats.languages[tweet.lang] = (stats.languages[tweet.lang] || 0) + 1;
                    }
                    
                    // 国家统计
                    if (tweet.country) {
                        stats.countries[tweet.country] = (stats.countries[tweet.country] || 0) + 1;
                    }
                    
                    // 时区统计
                    if (tweet.time_zone) {
                        stats.timezones[tweet.time_zone] = (stats.timezones[tweet.time_zone] || 0) + 1;
                    }
                    
                    // 来源统计
                    if (tweet.source) {
                        let source = tweet.source.replace(/<[^>]*>/g, '');
                        stats.sources[source] = (stats.sources[source] || 0) + 1;
                    }
                    
                    // 地点类型统计
                    if (tweet.place_type) {
                        stats.placeTypes[tweet.place_type] = (stats.placeTypes[tweet.place_type] || 0) + 1;
                    }
                    
                    // 文本长度
                    if (tweet.text) {
                        stats.textLengths.push(tweet.text.length);
                    }
                    
                    // 粉丝数量
                    if (tweet.followers_count) {
                        const count = parseInt(tweet.followers_count);
                        if (!isNaN(count)) {
                            stats.followerCounts.push(count);
                        }
                    }
                    
                    // 地理位置
                    if (tweet.coordinates || tweet.geo) {
                        stats.hasGeo++;
                        if (tweet.coordinates) {
                            try {
                                const coords = JSON.parse(tweet.coordinates);
                                stats.coordinates.push(coords);
                            } catch (e) {}
                        }
                    }
                    
                    // 标签和链接
                    if (tweet.hashtags && tweet.hashtags !== '[]') {
                        stats.hasHashtags++;
                    }
                    
                    if (tweet.urls && tweet.urls !== '[]') {
                        stats.hasUrls++;
                    }
                    
                    // 用户统计
                    if (tweet.screen_name) {
                        stats.uniqueUsers.add(tweet.screen_name);
                    }
                    
                    // 时间分布
                    if (tweet.created_at) {
                        const date = new Date(tweet.created_at);
                        const hour = date.getHours();
                        stats.timeDistribution[hour] = (stats.timeDistribution[hour] || 0) + 1;
                    }
                });
                
                processedCount = endIndex;
                const progress = Math.round((processedCount / totalTweets) * 100);
                
                // 发送进度更新
                self.postMessage({
                    type: 'progress',
                    progress,
                    processedCount,
                    totalTweets
                });
                
                if (processedCount < totalTweets) {
                    setTimeout(processBatch, 0);
                } else {
                    // 将 Set 转换为数组以便传递
                    const uniqueUsersArray = Array.from(stats.uniqueUsers);
                    stats.uniqueUsers = uniqueUsersArray;
                    
                    // 处理完成，发送结果
                    self.postMessage({
                        type: 'complete',
                        stats
                    });
                }
            }
            
            // 开始处理第一批
            processBatch();
        };
    `;
    
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    const workerUrl = URL.createObjectURL(blob);
    analysisWorker = new Worker(workerUrl);
    
    analysisWorker.onmessage = function(e) {
        const { type, progress, processedCount, totalTweets, stats } = e.data;
        
        if (type === 'progress') {
            updateAnalysisProgress(progress);
            showAnalysisStatus(`已处理 ${processedCount.toLocaleString()} / ${totalTweets.toLocaleString()} 条推文 (${progress}%)`);
            updateMemoryUsage();
        } else if (type === 'complete') {
            hideAnalysisProgress();
            displayResults(stats);
            isAnalyzing = false;
            showSuccess("分析完成！");
            enableAnalyzeButton();
            
            // 清理worker
            URL.revokeObjectURL(workerUrl);
        }
    };
    
    analysisWorker.onerror = function(error) {
        console.error('Worker error:', error);
        showError('分析过程中发生错误: ' + error.message);
        isAnalyzing = false;
        hideAnalysisProgress();
        enableAnalyzeButton();
    };
    
    return analysisWorker;
}

function loadSampleData() {
    // 删除示例数据，使用空的样本数据
    const sampleData = `[]`;
    document.getElementById('jsonInput').value = sampleData;
}

function clearData() {
    document.getElementById('jsonInput').value = '';
    document.getElementById('results').style.display = 'none';
    document.getElementById('error').style.display = 'none';
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('uploadProgress').style.display = 'none';
    document.getElementById('workerStatus').style.display = 'none';
    document.getElementById('fileInput').value = '';
    
    if (analysisWorker) {
        analysisWorker.terminate();
        analysisWorker = null;
    }
    
    if (fileReader) {
        fileReader.abort();
        fileReader = null;
    }
    
    isAnalyzing = false;
    enableAnalyzeButton();
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.innerHTML = `<div class="error">❌ ${message}</div>`;
    errorDiv.style.display = 'block';
}

function showSuccess(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.innerHTML = `<div style="background: #f0f9ff; border: 1px solid #0ea5e9; color: #0c4a6e; padding: 15px; border-radius: 10px; margin: 10px 0;">✅ ${message}</div>`;
    errorDiv.style.display = 'block';
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 3000);
}

function showWarning(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.innerHTML = `<div style="background: #fffbeb; border: 1px solid #f59e0b; color: #92400e; padding: 15px; border-radius: 10px; margin: 10px 0;">⚠️ ${message}</div>`;
    errorDiv.style.display = 'block';
}

function analyzeData() {
    if (isAnalyzing) {
        showWarning('正在分析中，请稍候...');
        return;
    }
    
    const input = document.getElementById('jsonInput').value.trim();
    if (!input) {
        showError('请输入JSON数据');
        return;
    }
    
    // 禁用分析按钮
    disableAnalyzeButton();
    isAnalyzing = true;
    
    // 显示分析进度
    showAnalysisProgress();
    
    // 使用 setTimeout 让UI有时间更新
    setTimeout(() => {
        try {
            showAnalysisStatus('正在解析JSON数据...');
            let data = JSON.parse(input);
            if (!Array.isArray(data)) {
                data = [data];
            }
            
            tweetData = data;
            showAnalysisStatus(`解析完成！共 ${data.length} 条推文，开始分析...`);
            
            // 创建并启动Web Worker进行分析
            const worker = createAnalysisWorker();
            worker.postMessage({
                data: tweetData,
                batchSize: 1000
            });
            
            document.getElementById('error').style.display = 'none';
            
        } catch (e) {
            hideAnalysisProgress();
            showError('JSON格式错误: ' + e.message);
            isAnalyzing = false;
            enableAnalyzeButton();
        }
    }, 100);
}

function showAnalysisProgress() {
    const progressHtml = `
        <div id="analysisProgress" style="margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 15px; text-align: center;">
            <div style="font-size: 1.2em; font-weight: 600; color: #495057; margin-bottom: 15px;">
                🔄 正在分析数据...
            </div>
            <div class="progress-bar" style="height: 12px; margin-bottom: 10px;">
                <div id="analysisProgressFill" class="progress-fill" style="width: 0%; animation: pulse 2s infinite;"></div>
            </div>
            <div id="analysisStatus" style="color: #666; font-size: 0.9em;">
                准备开始...
            </div>
            <div style="margin-top: 15px; text-align: right; font-size: 0.8em; color: #666;">
                <span id="analysisMemory">内存使用: 计算中...</span>
            </div>
        </div>
    `;
    
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = progressHtml;
    resultsDiv.style.display = 'block';
    
    // 显示Worker状态
    const workerStatus = document.getElementById('workerStatus');
    workerStatus.textContent = '🔄 多线程分析已启动';
    workerStatus.style.display = 'block';
}

function showAnalysisStatus(message) {
    const statusElement = document.getElementById('analysisStatus');
    if (statusElement) {
        statusElement.textContent = message;
    }
}

function updateAnalysisProgress(percent) {
    const progressFill = document.getElementById('analysisProgressFill');
    if (progressFill) {
        progressFill.style.width = percent + '%';
        progressFill.style.animation = percent < 100 ? 'pulse 2s infinite' : 'none';
    }
}

function hideAnalysisProgress() {
    const analysisProgress = document.getElementById('analysisProgress');
    if (analysisProgress) {
        analysisProgress.remove();
    }
    
    // 隐藏Worker状态
    setTimeout(() => {
        const workerStatus = document.getElementById('workerStatus');
        workerStatus.style.display = 'none';
    }, 1000);
}

function updateMemoryUsage() {
    // 如果浏览器支持性能API
    if (window.performance && window.performance.memory) {
        const memory = window.performance.memory;
        const memoryUsed = Math.round(memory.usedJSHeapSize / (1024 * 1024));
        const memoryTotal = Math.round(memory.jsHeapSizeLimit / (1024 * 1024));
        
        const memoryElement = document.getElementById('analysisMemory');
        if (memoryElement) {
            memoryElement.textContent = `内存使用: ${memoryUsed} MB / ${memoryTotal} MB`;
        }
    }
}

function disableAnalyzeButton() {
    const button = document.getElementById('analyzeBtn');
    button.disabled = true;
    button.textContent = '🔄 分析中...';
}

function enableAnalyzeButton() {
    const button = document.getElementById('analyzeBtn');
    button.disabled = false;
    button.textContent = '🔍 开始分析';
}

function displayResults(stats) {
    const resultsDiv = document.getElementById('results');
    
    // 计算统计值
    const avgTextLength = stats.textLengths.length > 0 ? 
        Math.round(stats.textLengths.reduce((a, b) => a + b, 0) / stats.textLengths.length) : 0;
    
    const avgFollowers = stats.followerCounts.length > 0 ? 
        Math.round(stats.followerCounts.reduce((a, b) => a + b, 0) / stats.followerCounts.length) : 0;
    
    let html = `
        <div class="summary">
            <h2>📋 数据概览</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <span class="summary-number">${stats.total}</span>
                    <span class="summary-label">总推文数</span>
                </div>
                <div class="summary-item">
                    <span class="summary-number">${stats.uniqueUsers.length}</span>
                    <span class="summary-label">独立用户</span>
                </div>
                <div class="summary-item">
                    <span class="summary-number">${Object.keys(stats.languages).length}</span>
                    <span class="summary-label">语言类型</span>
                </div>
                <div class="summary-item">
                    <span class="summary-number">${Object.keys(stats.countries).length}</span>
                    <span class="summary-label">国家数量</span>
                </div>
                <div class="summary-item">
                    <span class="summary-number">${avgTextLength}</span>
                    <span class="summary-label">平均字符数</span>
                </div>
                <div class="summary-item">
                    <span class="summary-number">${Math.round((stats.hasGeo / stats.total) * 100)}%</span>
                    <span class="summary-label">有地理位置</span>
                </div>
            </div>
        </div>
        
        <div class="stats-grid">
    `;
    
    // 语言分布
    html += createStatCard('🌐 语言分布', stats.languages, stats.total);
    
    // 国家分布
    html += createStatCard('🌍 国家分布', stats.countries, stats.total);
    
    // 时区分布
    html += createStatCard('🕐 时区分布', stats.timezones, stats.total);
    
    // 来源分布
    html += createStatCard('📱 发布来源', stats.sources, stats.total);
    
    // 地点类型
    if (Object.keys(stats.placeTypes).length > 0) {
        html += createStatCard('📍 地点类型', stats.placeTypes, stats.total);
    }
    
    // 文本统计
    if (stats.textLengths.length > 0) {
        const textStats = {
            '平均长度': avgTextLength,
            '最短': Math.min(...stats.textLengths),
            '最长': Math.max(...stats.textLengths),
            '总字符数': stats.textLengths.reduce((a, b) => a + b, 0)
        };
        html += createStatCard('📝 文本统计', textStats);
    }
    
    // 粉丝统计
    if (stats.followerCounts.length > 0) {
        const followerStats = {
            '平均粉丝数': avgFollowers,
            '最少粉丝': Math.min(...stats.followerCounts),
            '最多粉丝': Math.max(...stats.followerCounts),
            '中位数': getMedian(stats.followerCounts)
        };
        html += createStatCard('👥 粉丝统计', followerStats);
    }
    
    // 内容特征
    const contentStats = {
        '包含地理位置': `${stats.hasGeo} (${Math.round((stats.hasGeo / stats.total) * 100)}%)`,
        '包含标签': `${stats.hasHashtags} (${Math.round((stats.hasHashtags / stats.total) * 100)}%)`,
        '包含链接': `${stats.hasUrls} (${Math.round((stats.hasUrls / stats.total) * 100)}%)`
    };
    html += createStatCard('🏷️ 内容特征', contentStats);
    
    // 时间分布
    if (Object.keys(stats.timeDistribution).length > 0) {
        html += createStatCard('⏰ 发布时间分布（24小时制）', stats.timeDistribution, stats.total);
    }
    
    html += '</div>';
    
    resultsDiv.innerHTML = html;
    resultsDiv.style.display = 'block';
}

function createStatCard(title, data, total = null) {
    let html = `
        <div class="stat-card">
            <div class="stat-title">${title}</div>
    `;
    
    if (typeof data === 'object') {
        const sortedEntries = Object.entries(data)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 10); // 显示前10项
        
        sortedEntries.forEach(([key, value]) => {
            const percentage = total ? Math.round((value / total) * 100) : null;
            html += `
                <div class="stat-item">
                    <span class="stat-label">${key}</span>
                    <span class="stat-value">${value}${percentage ? ` (${percentage}%)` : ''}</span>
                </div>
            `;
            
            if (total && percentage) {
                html += `
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${percentage}%"></div>
                    </div>
                `;
            }
        });
        
        if (Object.keys(data).length > 10) {
            html += `<div class="stat-item"><span class="stat-label">... 还有 ${Object.keys(data).length - 10} 项</span></div>`;
        }
    }
    
    html += '</div>';
    return html;
}

function getMedian(arr) {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : Math.round((sorted[mid - 1] + sorted[mid]) / 2);
}

// 文件上传相关功能
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
    const memoryUsage = document.getElementById('memoryUsage');
    
    // 点击上传区域
    fileUploadZone.addEventListener('click', () => {
        fileInput.click();
    });
    
    // 文件选择
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    // 拖拽功能
    fileUploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileUploadZone.classList.add('dragover');
    });
    
    fileUploadZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        fileUploadZone.classList.remove('dragover');
    });
    
    fileUploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        fileUploadZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    function handleFileSelect(file) {
        if (!file) return;
        
        if (!file.name.toLowerCase().endsWith('.json')) {
            showError('请选择JSON格式的文件');
            return;
        }
        
        // 显示文件信息
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        fileInfo.style.display = 'block';
        
        // 大文件警告
        if (file.size > 50 * 1024 * 1024) { // 50MB
            showWarning(`文件较大 (${formatFileSize(file.size)})，读取可能需要一些时间，使用多线程分析...`);
        }
        
        // 显示进度条
        uploadProgress.style.display = 'block';
        progressText.textContent = '正在读取文件...';
        progressPercent.textContent = '0%';
        progressFill.style.width = '0%';
        totalSize.textContent = formatFileSize(file.size);
        processedSize.textContent = '0 MB';
        remainingTime.textContent = '计算中...';
        
        // 使用分块读取大文件
        readFileWithProgress(file);
    }
    
    function readFileWithProgress(file) {
        if (fileReader) {
            fileReader.abort();
        }
        
        const chunkSize = 2 * 1024 * 1024; // 2MB chunks
        const totalChunks = Math.ceil(file.size / chunkSize);
        let currentChunk = 0;
        let result = '';
        const startTime = Date.now();
        
        fileReader = new FileReader();
        
        function readNextChunk() {
            const start = currentChunk * chunkSize;
            const end = Math.min(start + chunkSize, file.size);
            const chunk = file.slice(start, end);
            
            fileReader.onload = (e) => {
                result += e.target.result;
                currentChunk++;
                
                // 更新进度
                const progress = (currentChunk / totalChunks) * 100;
                const processedBytes = Math.min(currentChunk * chunkSize, file.size);
                
                progressFill.style.width = progress + '%';
                progressPercent.textContent = Math.round(progress) + '%';
                processedSize.textContent = formatFileSize(processedBytes);
                
                // 更新内存使用
                if (window.performance && window.performance.memory) {
                    const memory = window.performance.memory;
                    const memoryUsed = Math.round(memory.usedJSHeapSize / (1024 * 1024));
                    const memoryTotal = Math.round(memory.jsHeapSizeLimit / (1024 * 1024));
                    memoryUsage.textContent = `${memoryUsed} MB / ${memoryTotal} MB`;
                }
                
                // 计算剩余时间
                if (currentChunk > 1) {
                    const elapsed = Date.now() - startTime;
                    const rate = processedBytes / elapsed; // bytes per ms
                    const remaining = (file.size - processedBytes) / rate;
                    remainingTime.textContent = formatTime(remaining);
                }
                
                if (currentChunk < totalChunks) {
                    // 使用 setTimeout 避免阻塞 UI
                    setTimeout(readNextChunk, 10);
                } else {
                    // 读取完成
                    progressText.textContent = '文件读取完成！';
                    document.getElementById('jsonInput').value = result;
                    
                    setTimeout(() => {
                        uploadProgress.style.display = 'none';
                        showSuccess(`文件 "${file.name}" 已成功加载！(${formatFileSize(file.size)})`);
                        // 自动开始分析
                        analyzeData();
                    }, 500);
                }
            };
            
            fileReader.onerror = () => {
                showError('文件读取失败');
                uploadProgress.style.display = 'none';
            };
            
            fileReader.readAsText(chunk);
        }
        
        readNextChunk();
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
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

function loadSampleData() {
    // 使用空的样本数据
    const sampleData = `[]`;
    document.getElementById('jsonInput').value = sampleData;
}

// 页面加载时初始化
window.onload = function() {
    setupFileUpload();
};