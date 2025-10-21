/**
 * Logger Utility
 * 
 * Provides centralized logging for the EXO WebGPU browser node
 * with different log levels and output targets.
 */

export class Logger {
    constructor(options = {}) {
        this.config = {
            level: options.level || 'info',
            enableConsole: options.enableConsole !== false,
            enableUI: options.enableUI !== false,
            maxEntries: options.maxEntries || 1000,
            ...options
        };
        
        this.levels = {
            debug: 0,
            info: 1,
            warn: 2,
            error: 3
        };
        
        this.entries = [];
        this.uiManager = null;
    }
    
    setUIManager(uiManager) {
        this.uiManager = uiManager;
    }
    
    log(message, ...args) {
        this.write('info', message, ...args);
    }
    
    info(message, ...args) {
        this.write('info', message, ...args);
    }
    
    warn(message, ...args) {
        this.write('warn', message, ...args);
    }
    
    error(message, ...args) {
        this.write('error', message, ...args);
    }
    
    debug(message, ...args) {
        this.write('debug', message, ...args);
    }
    
    write(level, message, ...args) {
        const levelValue = this.levels[level] || 0;
        const configLevelValue = this.levels[this.config.level] || 0;
        
        if (levelValue < configLevelValue) {
            return; // Skip if below configured level
        }
        
        const timestamp = new Date().toISOString();
        const formattedMessage = this.formatMessage(message, ...args);
        
        const entry = {
            timestamp,
            level,
            message: formattedMessage,
            args: args
        };
        
        // Store entry
        this.entries.push(entry);
        
        // Limit entries
        if (this.entries.length > this.config.maxEntries) {
            this.entries = this.entries.slice(-this.config.maxEntries);
        }
        
        // Output to console
        if (this.config.enableConsole) {
            this.writeToConsole(level, formattedMessage, ...args);
        }
        
        // Output to UI
        if (this.config.enableUI && this.uiManager) {
            this.uiManager.addLogMessage(formattedMessage, level);
        }
    }
    
    formatMessage(message, ...args) {
        if (typeof message === 'object') {
            message = JSON.stringify(message, null, 2);
        }
        
        if (args.length > 0) {
            // Simple string interpolation for common cases
            let formatted = message;
            args.forEach((arg, index) => {
                const placeholder = `{${index}}`;
                if (formatted.includes(placeholder)) {
                    formatted = formatted.replace(placeholder, this.formatValue(arg));
                } else if (index === 0 && !formatted.includes('{')) {
                    // Append first arg if no placeholders
                    formatted += ` ${this.formatValue(arg)}`;
                }
            });
            return formatted;
        }
        
        return message;
    }
    
    formatValue(value) {
        if (value === null) return 'null';
        if (value === undefined) return 'undefined';
        if (typeof value === 'object') {
            try {
                return JSON.stringify(value);
            } catch (e) {
                return '[Object]';
            }
        }
        return String(value);
    }
    
    writeToConsole(level, message, ...args) {
        const timestamp = new Date().toLocaleTimeString();
        const prefix = `[${timestamp}] [${level.toUpperCase()}]`;
        
        switch (level) {
            case 'debug':
                console.debug(prefix, message, ...args);
                break;
            case 'info':
                console.info(prefix, message, ...args);
                break;
            case 'warn':
                console.warn(prefix, message, ...args);
                break;
            case 'error':
                console.error(prefix, message, ...args);
                break;
            default:
                console.log(prefix, message, ...args);
        }
    }
    
    clear() {
        this.entries = [];
        
        if (this.config.enableConsole) {
            console.clear();
        }
        
        if (this.config.enableUI && this.uiManager) {
            this.uiManager.clearLog();
        }
    }
    
    getEntries(level = null, limit = null) {
        let entries = this.entries;
        
        if (level) {
            entries = entries.filter(entry => entry.level === level);
        }
        
        if (limit) {
            entries = entries.slice(-limit);
        }
        
        return entries;
    }
    
    exportLogs() {
        const logs = this.entries.map(entry => ({
            timestamp: entry.timestamp,
            level: entry.level,
            message: entry.message
        }));
        
        const blob = new Blob([JSON.stringify(logs, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `exo-node-logs-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.info('📥 Logs exported to file');
    }
    
    setLevel(level) {
        if (level in this.levels) {
            this.config.level = level;
            this.info(`🔧 Log level set to ${level}`);
        } else {
            this.warn(`⚠️ Invalid log level: ${level}`);
        }
    }
    
    getLevel() {
        return this.config.level;
    }
    
    // Performance logging
    time(label) {
        this.timeStamps = this.timeStamps || new Map();
        this.timeStamps.set(label, performance.now());
        this.debug(`⏱️ Timer started: ${label}`);
    }
    
    timeEnd(label) {
        if (!this.timeStamps || !this.timeStamps.has(label)) {
            this.warn(`⚠️ Timer not found: ${label}`);
            return;
        }
        
        const startTime = this.timeStamps.get(label);
        const duration = performance.now() - startTime;
        this.timeStamps.delete(label);
        
        this.info(`⏱️ ${label}: ${duration.toFixed(2)}ms`);
        return duration;
    }
    
    // Memory usage logging
    logMemoryUsage() {
        if ('memory' in performance) {
            const memory = performance.memory;
            this.info(`💾 Memory usage:`, {
                used: `${(memory.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB`,
                total: `${(memory.totalJSHeapSize / 1024 / 1024).toFixed(2)} MB`,
                limit: `${(memory.jsHeapSizeLimit / 1024 / 1024).toFixed(2)} MB`
            });
        } else {
            this.debug('💾 Memory API not available');
        }
    }
    
    // Network logging helpers
    logNetworkEvent(event, details) {
        this.info(`🌐 Network: ${event}`, details);
    }
    
    logPeerEvent(event, peerId, details = {}) {
        this.info(`👤 Peer ${event}: ${this.shortenPeerId(peerId)}`, details);
    }
    
    logInferenceEvent(event, details) {
        this.info(`🧠 Inference: ${event}`, details);
    }
    
    shortenPeerId(peerId) {
        if (!peerId || peerId.length <= 16) return peerId;
        return `${peerId.substring(0, 8)}...${peerId.substring(peerId.length - 8)}`;
    }
    
    // Statistics
    getStatistics() {
        const stats = {
            totalEntries: this.entries.length,
            byLevel: {},
            recentActivity: {}
        };
        
        // Count by level
        this.entries.forEach(entry => {
            stats.byLevel[entry.level] = (stats.byLevel[entry.level] || 0) + 1;
        });
        
        // Recent activity (last hour)
        const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
        const recentEntries = this.entries.filter(entry => 
            new Date(entry.timestamp) > oneHourAgo
        );
        
        stats.recentActivity.total = recentEntries.length;
        recentEntries.forEach(entry => {
            stats.recentActivity[entry.level] = (stats.recentActivity[entry.level] || 0) + 1;
        });
        
        return stats;
    }
    
    // Debug helpers
    logObjectProperties(obj, label = 'Object') {
        if (typeof obj !== 'object' || obj === null) {
            this.debug(`${label}: ${obj}`);
            return;
        }
        
        const properties = Object.keys(obj).map(key => {
            const value = obj[key];
            const type = typeof value;
            return `${key}: ${type === 'function' ? '[Function]' : value}`;
        });
        
        this.debug(`${label} properties:`, properties);
    }
    
    logError(error, context = '') {
        const errorInfo = {
            message: error.message,
            stack: error.stack,
            name: error.name,
            context: context
        };
        
        this.error(`❌ Error${context ? ` in ${context}` : ''}:`, errorInfo);
    }
    
    logPerformanceEntry(entry) {
        this.debug(`⚡ Performance: ${entry.name}`, {
            duration: `${entry.duration.toFixed(2)}ms`,
            startTime: `${entry.startTime.toFixed(2)}ms`,
            entryType: entry.entryType
        });
    }
}
