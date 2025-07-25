/**
 * UI Manager
 * 
 * Handles all user interface updates and interactions
 * for the EXO WebGPU browser node.
 */

export class UIManager {
    constructor() {
        this.elements = this.initializeElements();
        this.formatters = this.initializeFormatters();
    }
    
    initializeElements() {
        return {
            // Status elements
            statusIndicator: document.getElementById('statusIndicator'),
            statusText: document.getElementById('statusText'),
            webgpuIndicator: document.getElementById('webgpuIndicator'),
            webgpuStatus: document.getElementById('webgpuStatus'),
            
            // Metric elements
            peerCount: document.getElementById('peerCount'),
            taskCount: document.getElementById('taskCount'),
            uptime: document.getElementById('uptime'),
            inferenceTime: document.getElementById('inferenceTime'),
            
            // Control elements
            startBtn: document.getElementById('startBtn'),
            stopBtn: document.getElementById('stopBtn'),
            bootstrapInput: document.getElementById('bootstrapInput'),
            
            // Network elements
            myPeerId: document.getElementById('myPeerId'),
            peerList: document.getElementById('peerList'),
            
            // Share elements
            shareUrl: document.getElementById('shareUrl'),
            
            // Log element
            logOutput: document.getElementById('logOutput')
        };
    }
    
    initializeFormatters() {
        return {
            time: (ms) => {
                if (ms < 1000) return `${ms}ms`;
                if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
                if (ms < 3600000) return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
                return `${Math.floor(ms / 3600000)}h ${Math.floor((ms % 3600000) / 60000)}m`;
            },
            
            peerId: (id) => {
                if (!id || id === 'Not started') return id;
                return id.length > 20 ? `${id.substring(0, 8)}...${id.substring(id.length - 8)}` : id;
            },
            
            number: (num) => {
                return num.toLocaleString();
            }
        };
    }
    
    updateStatus(status, text) {
        if (!this.elements.statusIndicator || !this.elements.statusText) return;
        
        // Remove existing status classes
        this.elements.statusIndicator.className = 'status-indicator';
        
        // Add new status class
        switch (status) {
            case 'connecting':
                this.elements.statusIndicator.classList.add('status-connecting');
                break;
            case 'connected':
                this.elements.statusIndicator.classList.add('status-connected');
                break;
            case 'error':
                this.elements.statusIndicator.classList.add('status-error');
                break;
            default:
                this.elements.statusIndicator.classList.add('status-connecting');
        }
        
        this.elements.statusText.textContent = text;
    }
    
    updateWebGPUStatus(supportInfo) {
        if (!this.elements.webgpuIndicator || !this.elements.webgpuStatus) return;
        
        this.elements.webgpuIndicator.className = 'status-indicator';
        
        if (supportInfo.supported) {
            this.elements.webgpuIndicator.classList.add('status-connected');
            this.elements.webgpuStatus.textContent = 'WebGPU available and ready';
        } else if (supportInfo.fallback) {
            this.elements.webgpuIndicator.classList.add('status-connecting');
            this.elements.webgpuStatus.textContent = 'WebGL fallback available';
        } else {
            this.elements.webgpuIndicator.classList.add('status-error');
            this.elements.webgpuStatus.textContent = 'No GPU acceleration available';
        }
    }
    
    updatePeerCount(count) {
        if (this.elements.peerCount) {
            this.elements.peerCount.textContent = this.formatters.number(count);
        }
    }
    
    updateTaskCount(count) {
        if (this.elements.taskCount) {
            this.elements.taskCount.textContent = this.formatters.number(count);
        }
    }
    
    updateUptime(ms) {
        if (this.elements.uptime) {
            this.elements.uptime.textContent = this.formatters.time(ms);
        }
    }
    
    updateInferenceTime(ms) {
        if (this.elements.inferenceTime) {
            this.elements.inferenceTime.textContent = `${ms}ms`;
        }
    }
    
    updatePeerId(peerId) {
        if (this.elements.myPeerId) {
            this.elements.myPeerId.textContent = this.formatters.peerId(peerId);
            this.elements.myPeerId.title = peerId; // Full ID on hover
        }
    }
    
    updatePeerList(peers) {
        if (!this.elements.peerList) return;
        
        if (peers.length === 0) {
            this.elements.peerList.innerHTML = `
                <div style="opacity: 0.6; text-align: center; padding: 1rem;">
                    No peers connected
                </div>
            `;
            return;
        }
        
        const peerElements = peers.map(peer => {
            const peerId = peer.id || peer;
            const connectedTime = peer.connectedAt ? 
                this.formatters.time(Date.now() - peer.connectedAt) : 'Unknown';
                
            return `
                <div class="peer">
                    <div>
                        <div style="font-weight: 500;">${this.formatters.peerId(peerId)}</div>
                        <div class="peer-id">Connected: ${connectedTime}</div>
                    </div>
                    <div style="font-size: 0.8rem; opacity: 0.7;">
                        ${peer.capabilities ? '🎯' : '⏳'}
                    </div>
                </div>
            `;
        });
        
        this.elements.peerList.innerHTML = peerElements.join('');
    }
    
    setNodeRunning(isRunning) {
        if (this.elements.startBtn) {
            this.elements.startBtn.disabled = isRunning;
        }
        
        if (this.elements.stopBtn) {
            this.elements.stopBtn.disabled = !isRunning;
        }
    }
    
    getBootstrapNode() {
        return this.elements.bootstrapInput ? this.elements.bootstrapInput.value : '';
    }
    
    setBootstrapNode(address) {
        if (this.elements.bootstrapInput) {
            this.elements.bootstrapInput.value = address;
        }
    }
    
    updateShareUrl(url) {
        if (this.elements.shareUrl) {
            this.elements.shareUrl.textContent = url;
        }
    }
    
    getShareUrl() {
        return this.elements.shareUrl ? this.elements.shareUrl.textContent : '';
    }
    
    addLogMessage(message, type = 'info') {
        if (!this.elements.logOutput) return;
        
        const timestamp = new Date().toLocaleTimeString();
        const prefix = this.getLogPrefix(type);
        const logLine = `[${timestamp}] ${prefix} ${message}\n`;
        
        this.elements.logOutput.textContent += logLine;
        
        // Auto-scroll to bottom
        this.elements.logOutput.scrollTop = this.elements.logOutput.scrollHeight;
        
        // Limit log size to prevent memory issues
        this.limitLogSize();
    }
    
    getLogPrefix(type) {
        switch (type) {
            case 'info': return 'ℹ️';
            case 'success': return '✅';
            case 'warning': return '⚠️';
            case 'error': return '❌';
            case 'debug': return '🐛';
            default: return 'ℹ️';
        }
    }
    
    limitLogSize() {
        if (!this.elements.logOutput) return;
        
        const lines = this.elements.logOutput.textContent.split('\n');
        const maxLines = 1000;
        
        if (lines.length > maxLines) {
            const truncatedLines = lines.slice(-maxLines);
            this.elements.logOutput.textContent = truncatedLines.join('\n');
        }
    }
    
    clearLog() {
        if (this.elements.logOutput) {
            this.elements.logOutput.textContent = 'EXO WebGPU Browser Node ready...\n';
        }
    }
    
    showNotification(message, type = 'info', duration = 5000) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            z-index: 1000;
            max-width: 400px;
            word-wrap: break-word;
            border-left: 4px solid ${this.getNotificationColor(type)};
            animation: slideIn 0.3s ease-out;
        `;
        
        notification.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span>${this.getLogPrefix(type)}</span>
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" 
                        style="margin-left: auto; background: none; border: none; color: white; cursor: pointer; font-size: 1.2rem;">×</button>
            </div>
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Auto-remove after duration
        setTimeout(() => {
            if (notification.parentElement) {
                notification.style.animation = 'slideOut 0.3s ease-in forwards';
                setTimeout(() => notification.remove(), 300);
            }
        }, duration);
        
        // Add CSS animations if not already present
        this.addNotificationStyles();
    }
    
    getNotificationColor(type) {
        switch (type) {
            case 'success': return '#10b981';
            case 'warning': return '#fbbf24';
            case 'error': return '#ef4444';
            default: return '#3b82f6';
        }
    }
    
    addNotificationStyles() {
        if (document.getElementById('notification-styles')) return;
        
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            
            @keyframes slideOut {
                from {
                    transform: translateX(0);
                    opacity: 1;
                }
                to {
                    transform: translateX(100%);
                    opacity: 0;
                }
            }
        `;
        
        document.head.appendChild(style);
    }
    
    updateConnectionQuality(peerId, quality) {
        // Update peer connection quality indicators
        const peerElements = this.elements.peerList.querySelectorAll('.peer');
        
        peerElements.forEach(element => {
            const peerIdElement = element.querySelector('.peer-id');
            if (peerIdElement && peerIdElement.textContent.includes(this.formatters.peerId(peerId))) {
                const qualityIndicator = element.querySelector('.quality-indicator') || 
                    document.createElement('div');
                
                qualityIndicator.className = 'quality-indicator';
                qualityIndicator.style.cssText = `
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: ${this.getQualityColor(quality)};
                    margin-left: auto;
                `;
                qualityIndicator.title = `Connection quality: ${quality}`;
                
                if (!element.querySelector('.quality-indicator')) {
                    element.appendChild(qualityIndicator);
                }
            }
        });
    }
    
    getQualityColor(quality) {
        if (quality >= 0.8) return '#10b981'; // Good - green
        if (quality >= 0.5) return '#fbbf24'; // Fair - yellow
        return '#ef4444'; // Poor - red
    }
    
    setTheme(theme) {
        document.body.className = `theme-${theme}`;
        localStorage.setItem('exo-theme', theme);
    }
    
    getTheme() {
        return localStorage.getItem('exo-theme') || 'dark';
    }
    
    initializeTheme() {
        this.setTheme(this.getTheme());
    }
    
    // Utility methods for dynamic UI updates
    createElement(tag, className, content) {
        const element = document.createElement(tag);
        if (className) element.className = className;
        if (content) element.textContent = content;
        return element;
    }
    
    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    animateValue(element, start, end, duration = 1000) {
        if (!element) return;
        
        const startTime = performance.now();
        const range = end - start;
        
        const updateValue = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const current = start + (range * this.easeOutCubic(progress));
            element.textContent = Math.round(current);
            
            if (progress < 1) {
                requestAnimationFrame(updateValue);
            }
        };
        
        requestAnimationFrame(updateValue);
    }
    
    easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }
    
    // Performance monitoring
    startPerformanceMonitoring() {
        this.performanceMetrics = {
            startTime: performance.now(),
            frameCount: 0,
            lastFrameTime: performance.now()
        };
        
        const updateFPS = () => {
            this.performanceMetrics.frameCount++;
            const currentTime = performance.now();
            
            if (currentTime - this.performanceMetrics.lastFrameTime >= 1000) {
                const fps = this.performanceMetrics.frameCount;
                this.performanceMetrics.frameCount = 0;
                this.performanceMetrics.lastFrameTime = currentTime;
                
                // Update FPS display if element exists
                const fpsElement = document.getElementById('fps');
                if (fpsElement) {
                    fpsElement.textContent = `${fps} FPS`;
                }
            }
            
            requestAnimationFrame(updateFPS);
        };
        
        requestAnimationFrame(updateFPS);
    }
}
