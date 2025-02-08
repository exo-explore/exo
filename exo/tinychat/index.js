document.addEventListener("alpine:init", () => {
  Alpine.data("state", () => ({
    // current state
    cstate: {
      time: null,
      messages: [],
      selectedModel: 'llama-3.2-1b',
    },

    // historical state
    histories: JSON.parse(localStorage.getItem("histories")) || [],

    home: 0,
    generating: false,
    endpoint: `${window.location.origin}/v1`,

    // Initialize error message structure
    errorMessage: null,
    errorExpanded: false,
    errorTimeout: null,

    // performance tracking
    time_till_first: 0,
    tokens_per_second: 0,
    total_tokens: 0,

    // image handling
    imagePreview: null,

    // download progress
    downloadProgress: null,
    downloadProgressInterval: null, // To keep track of the polling interval

    // Pending message storage
    pendingMessage: null,

    modelPoolInterval: null,

    // Add models state alongside existing state
    models: {},

    // Show only models available locally
    showDownloadedOnly: false,

    topology: null,
    topologyInterval: null,

    // Add these new properties
    expandedGroups: {},

    init() {
      // Clean up any pending messages
      localStorage.removeItem("pendingMessage");

      // Get initial model list
      this.fetchInitialModels();

      // Start polling for download progress
      this.startDownloadProgressPolling();

      // Start model polling with the new pattern
      this.startModelPolling();
    },

    async fetchInitialModels() {
      try {
        const response = await fetch(`${window.location.origin}/initial_models`);
        if (response.ok) {
          const initialModels = await response.json();
          this.models = initialModels;
        }
      } catch (error) {
        console.error('Error fetching initial models:', error);
      }
    },

    async startModelPolling() {
      while (true) {
        try {
          await this.populateSelector();
          // Wait 15 seconds before next poll
          await new Promise(resolve => setTimeout(resolve, 15000));
        } catch (error) {
          console.error('Model polling error:', error);
          // If there's an error, wait before retrying
          await new Promise(resolve => setTimeout(resolve, 15000));
        }
      }
    },

    async populateSelector() {
      return new Promise((resolve, reject) => {
        const evtSource = new EventSource(`${window.location.origin}/modelpool`);

        evtSource.onmessage = (event) => {
          if (event.data === "[DONE]") {
            evtSource.close();
            resolve();
            return;
          }

          const modelData = JSON.parse(event.data);
          // Update existing model data while preserving other properties
          Object.entries(modelData).forEach(([modelName, data]) => {
            if (this.models[modelName]) {
              this.models[modelName] = {
                ...this.models[modelName],
                ...data,
                loading: false
              };
            }
          });
        };

        evtSource.onerror = (error) => {
          console.error('EventSource failed:', error);
          evtSource.close();
          reject(error);
        };
      });
    },

    removeHistory(cstate) {
      const index = this.histories.findIndex((state) => {
        return state.time === cstate.time;
      });
      if (index !== -1) {
        this.histories.splice(index, 1);
        localStorage.setItem("histories", JSON.stringify(this.histories));
      }
    },

    clearAllHistory() {
      this.histories = [];
      localStorage.setItem("histories", JSON.stringify([]));
    },

    // Utility functions
    formatBytes(bytes) {
      if (bytes === 0) return '0 B';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    formatDuration(seconds) {
      if (seconds === null || seconds === undefined || isNaN(seconds)) return '';
      const h = Math.floor(seconds / 3600);
      const m = Math.floor((seconds % 3600) / 60);
      const s = Math.floor(seconds % 60);
      if (h > 0) return `${h}h ${m}m ${s}s`;
      if (m > 0) return `${m}m ${s}s`;
      return `${s}s`;
    },

    async handleImageUpload(event) {
      const file = event.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          this.imagePreview = e.target.result;
          this.imageUrl = e.target.result; // Store the image URL
          // Add image preview to the chat
          this.cstate.messages.push({
            role: "user",
            content: `![Uploaded Image](${this.imagePreview})`,
          });
        };
        reader.readAsDataURL(file);
      }
    },


    async handleSend() {
      try {
        const el = document.getElementById("input-form");
        const value = el.value.trim();
        if (!value && !this.imagePreview) return;

        if (this.generating) return;
        this.generating = true;
        if (this.home === 0) this.home = 1;

        // ensure that going back in history will go back to home
        window.history.pushState({}, "", "/");

        // add message to list
        if (value) {
          this.cstate.messages.push({ role: "user", content: value });
        }

        // clear textarea
        el.value = "";
        el.style.height = "auto";
        el.style.height = el.scrollHeight + "px";

        localStorage.setItem("pendingMessage", value);
        this.processMessage(value);
      } catch (error) {
        console.error('error', error);
        this.setError(error);
        this.generating = false;
      }
    },

    async processMessage(value) {
      try {
        // reset performance tracking
        const prefill_start = Date.now();
        let start_time = 0;
        let tokens = 0;
        this.tokens_per_second = 0;

        // prepare messages for API request
        let apiMessages = this.cstate.messages.map(msg => {
          if (msg.content.startsWith('![Uploaded Image]')) {
            return {
              role: "user",
              content: [
                {
                  type: "image_url",
                  image_url: {
                    url: this.imageUrl
                  }
                },
                {
                  type: "text",
                  text: value // Use the actual text the user typed
                }
              ]
            };
          } else {
            return {
              role: msg.role,
              content: msg.content
            };
          }
        });
        
        if (this.cstate.selectedModel === "stable-diffusion-2-1-base") {
          // Send a request to the image generation endpoint
          console.log(apiMessages[apiMessages.length - 1].content)
          console.log(this.cstate.selectedModel)  
          console.log(this.endpoint)
          const response = await fetch(`${this.endpoint}/image/generations`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              "model": 'stable-diffusion-2-1-base',
              "prompt": apiMessages[apiMessages.length - 1].content,
              "image_url": this.imageUrl
            }),
          });
      
          if (!response.ok) {
            throw new Error("Failed to fetch");
          }
          const reader = response.body.getReader();
          let done = false;
          let gottenFirstChunk = false;
  
          while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            const decoder = new TextDecoder();
  
            if (value) {
              // Assume non-binary data (text) comes first
              const chunk = decoder.decode(value, { stream: true });
              const parsed = JSON.parse(chunk);
              console.log(parsed)
  
              if (parsed.progress) {
                if (!gottenFirstChunk) {
                  this.cstate.messages.push({ role: "assistant", content: "" });
                  gottenFirstChunk = true;
                }
                this.cstate.messages[this.cstate.messages.length - 1].content = parsed.progress;
              }
              else if (parsed.images) {
                if (!gottenFirstChunk) {
                  this.cstate.messages.push({ role: "assistant", content: "" });
                  gottenFirstChunk = true;
                }
                const imageUrl = parsed.images[0].url;
                console.log(imageUrl)
                this.cstate.messages[this.cstate.messages.length - 1].content = `![Generated Image](${imageUrl}?t=${Date.now()})`;
              }
            }
          }
        }
        
        else{        
          const containsImage = apiMessages.some(msg => Array.isArray(msg.content) && msg.content.some(item => item.type === 'image_url'));
          if (containsImage) {
            // Map all messages with string content to object with type text
            apiMessages = apiMessages.map(msg => {
              if (typeof msg.content === 'string') {
                return {
                  ...msg,
                  content: [
                    {
                      type: "text",
                      text: msg.content
                    }
                  ]
                };
              }
              return msg;
            });
          }

          console.log(apiMessages)
          //start receiving server sent events
          let gottenFirstChunk = false;
          for await (
            const chunk of this.openaiChatCompletion(this.cstate.selectedModel, apiMessages)
          ) {
            if (!gottenFirstChunk) {
              this.cstate.messages.push({ role: "assistant", content: "" });
              gottenFirstChunk = true;
            }

            // add chunk to the last message
            this.cstate.messages[this.cstate.messages.length - 1].content += chunk;

            // calculate performance tracking
            tokens += 1;
            this.total_tokens += 1;
            if (start_time === 0) {
              start_time = Date.now();
              this.time_till_first = start_time - prefill_start;
            } else {
              const diff = Date.now() - start_time;
              if (diff > 0) {
                this.tokens_per_second = tokens / (diff / 1000);
              }
            }
          }
        }
        // Clean the cstate before adding it to histories
        const cleanedCstate = JSON.parse(JSON.stringify(this.cstate));
        cleanedCstate.messages = cleanedCstate.messages.map(msg => {
          if (Array.isArray(msg.content)) {
            return {
              ...msg,
              content: msg.content.map(item =>
                item.type === 'image_url' ? { type: 'image_url', image_url: { url: '[IMAGE_PLACEHOLDER]' } } : item
              )
            };
          }
          return msg;
        });

        // Update the state in histories or add it if it doesn't exist
        const index = this.histories.findIndex((cstate) => cstate.time === cleanedCstate.time);
        cleanedCstate.time = Date.now();
        if (index !== -1) {
          // Update the existing entry
          this.histories[index] = cleanedCstate;
        } else {
          // Add a new entry
          this.histories.push(cleanedCstate);
        }
        console.log(this.histories)
        // update in local storage
        try {
          localStorage.setItem("histories", JSON.stringify(this.histories));
        } catch (error) {
          console.error("Failed to save histories to localStorage:", error);
        }
      } catch (error) {
        console.error('error', error);
        this.setError(error);
      } finally {
        this.generating = false;
      }
    },

    async handleEnter(event) {
      // if shift is not pressed
      if (!event.shiftKey) {
        event.preventDefault();
        await this.handleSend();
      }
    },

    updateTotalTokens(messages) {
      fetch(`${this.endpoint}/chat/token/encode`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages }),
      }).then((response) => response.json()).then((data) => {
        this.total_tokens = data.length;
      }).catch(console.error);
    },

    async *openaiChatCompletion(model, messages) {
      const response = await fetch(`${this.endpoint}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          "model": model,
          "messages": messages,
          "stream": true,
        }),
      });
      if (!response.ok) {
        const errorResBody = await response.json()
        if (errorResBody?.detail) {
          throw new Error(`Failed to fetch completions: ${errorResBody.detail}`);
        } else {
          throw new Error("Failed to fetch completions: Unknown error");
        }
      }

      const reader = response.body.pipeThrough(new TextDecoderStream())
        .pipeThrough(new EventSourceParserStream()).getReader();
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        if (value.type === "event") {
          const json = JSON.parse(value.data);
          if (json.choices) {
            const choice = json.choices[0];
            if (choice.finish_reason === "stop") break;
            if (choice.delta.content) yield choice.delta.content;
          }
        }
      }
    },

    async fetchDownloadProgress() {
      try {
        const response = await fetch(`${this.endpoint}/download/progress`);
        if (response.ok) {
          const data = await response.json();
          const progressArray = Object.values(data);
          if (progressArray.length > 0) {
            this.downloadProgress = progressArray.map(progress => {
              // Check if download is complete
              if (progress.status === "complete") {
                return {
                  ...progress,
                  isComplete: true,
                  percentage: 100
                };
              } else if (progress.status === "failed") {
                return {
                  ...progress,
                  isComplete: false,
                  errorMessage: "Download failed"
                };
              } else {
                return {
                  ...progress,
                  isComplete: false,
                  downloaded_bytes_display: this.formatBytes(progress.downloaded_bytes),
                  total_bytes_display: this.formatBytes(progress.total_bytes),
                  overall_speed_display: progress.overall_speed ? this.formatBytes(progress.overall_speed) + '/s' : '',
                  overall_eta_display: progress.overall_eta ? this.formatDuration(progress.overall_eta) : '',
                  percentage: ((progress.downloaded_bytes / progress.total_bytes) * 100).toFixed(2)
                };
              }
            });
            const allComplete = this.downloadProgress.every(progress => progress.isComplete);
            if (allComplete) {
              // Check for pendingMessage
              const savedMessage = localStorage.getItem("pendingMessage");
              if (savedMessage) {
                // Clear pendingMessage
                localStorage.removeItem("pendingMessage");
                // Call processMessage() with savedMessage
                if (this.lastErrorMessage) {
                  await this.processMessage(savedMessage);
                }
              }
              this.lastErrorMessage = null;
              this.downloadProgress = null;
            }
          } else {
            // No ongoing download
            this.downloadProgress = null;
          }
        }
      } catch (error) {
        console.error("Error fetching download progress:", error);
        this.downloadProgress = null;
      }
    },

    startDownloadProgressPolling() {
      if (this.downloadProgressInterval) {
        // Already polling
        return;
      }
      this.fetchDownloadProgress(); // Fetch immediately
      this.downloadProgressInterval = setInterval(() => {
        this.fetchDownloadProgress();
      }, 1000); // Poll every second
    },

    // Add a helper method to set errors consistently
    setError(error) {
      this.errorMessage = {
        basic: error.message || "An unknown error occurred",
        stack: error.stack || ""
      };
      this.errorExpanded = false;

      if (this.errorTimeout) {
        clearTimeout(this.errorTimeout);
      }

      if (!this.errorExpanded) {
        this.errorTimeout = setTimeout(() => {
          this.errorMessage = null;
          this.errorExpanded = false;
        }, 30 * 1000);
      }
    },

    async deleteModel(modelName, model) {
      const downloadedSize = model.total_downloaded || 0;
      const sizeMessage = downloadedSize > 0 ?
        `This will free up ${this.formatBytes(downloadedSize)} of space.` :
        'This will remove any partially downloaded files.';

      if (!confirm(`Are you sure you want to delete ${model.name}? ${sizeMessage}`)) {
        return;
      }

      try {
        const response = await fetch(`${window.location.origin}/models/${modelName}`, {
          method: 'DELETE',
          headers: {
            'Content-Type': 'application/json'
          }
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Failed to delete model');
        }

        // Update the model status in the UI
        if (this.models[modelName]) {
          this.models[modelName].downloaded = false;
          this.models[modelName].download_percentage = 0;
          this.models[modelName].total_downloaded = 0;
        }

        // If this was the selected model, switch to a different one
        if (this.cstate.selectedModel === modelName) {
          const availableModel = Object.keys(this.models).find(key => this.models[key].downloaded);
          this.cstate.selectedModel = availableModel || 'llama-3.2-1b';
        }

        // Show success message
        console.log(`Model deleted successfully from: ${data.path}`);

        // Refresh the model list
        await this.populateSelector();
      } catch (error) {
        console.error('Error deleting model:', error);
        this.setError(error.message || 'Failed to delete model');
      }
    },

    async handleDownload(modelName) {
      try {
        const response = await fetch(`${window.location.origin}/download`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: modelName
          })
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || 'Failed to start download');
        }

        // Update the model's status immediately when download starts
        if (this.models[modelName]) {
          this.models[modelName] = {
            ...this.models[modelName],
            loading: true
          };
        }

      } catch (error) {
        console.error('Error starting download:', error);
        this.setError(error);
      }
    },

    async fetchTopology() {
      try {
        const response = await fetch(`${this.endpoint}/topology`);
        if (!response.ok) throw new Error('Failed to fetch topology');
        return await response.json();
      } catch (error) {
        console.error('Topology fetch error:', error);
        return null;
      }
    },

    initTopology() {
      // Initial fetch
      this.updateTopology();

      // Set up periodic updates
      this.topologyInterval = setInterval(() => this.updateTopology(), 5000);

      // Cleanup on page unload
      window.addEventListener('beforeunload', () => {
        if (this.topologyInterval) {
          clearInterval(this.topologyInterval);
        }
      });
    },

    async updateTopology() {
      const topologyData = await this.fetchTopology();
      if (!topologyData) return;

      const vizElement = this.$refs.topologyViz;
      vizElement.innerHTML = ''; // Clear existing visualization

      // Helper function to truncate node ID
      const truncateNodeId = (id) => id.substring(0, 8);

      // Create nodes from object
      Object.entries(topologyData.nodes).forEach(([nodeId, node]) => {
        const nodeElement = document.createElement('div');
        nodeElement.className = 'topology-node';

        // Get peer connections for this node
        const peerConnections = topologyData.peer_graph[nodeId] || [];
        const peerConnectionsHtml = peerConnections.map(peer => `
          <div class="peer-connection">
            <i class="fas fa-arrow-right"></i>
            <span>To ${truncateNodeId(peer.to_id)}: ${peer.description}</span>
          </div>
        `).join('');

        nodeElement.innerHTML = `
          <div class="node-info">
            <span class="status ${nodeId === topologyData.active_node_id ? 'active' : 'inactive'}"></span>
            <span>${node.model} [${truncateNodeId(nodeId)}]</span>
          </div>
          <div class="node-details">
            <span>${node.chip}</span>
            <span>${(node.memory / 1024).toFixed(1)}GB RAM</span>
            <span>${node.flops.fp32.toFixed(1)} TF</span>
          </div>
          <div class="peer-connections">
            ${peerConnectionsHtml}
          </div>
        `;
        vizElement.appendChild(nodeElement);
      });
    },

    // Add these helper methods
    countDownloadedModels(models) {
      return Object.values(models).filter(model => model.downloaded).length;
    },

    getGroupCounts(groupModels) {
      const total = Object.keys(groupModels).length;
      const downloaded = this.countDownloadedModels(groupModels);
      return `[${downloaded}/${total}]`;
    },

    // Update the existing groupModelsByPrefix method to include counts
    groupModelsByPrefix(models) {
      const groups = {};
      const filteredModels = this.showDownloadedOnly ?
        Object.fromEntries(Object.entries(models).filter(([, model]) => model.downloaded)) :
        models;

      Object.entries(filteredModels).forEach(([key, model]) => {
        const parts = key.split('-');
        const mainPrefix = parts[0].toUpperCase();
        
        let subPrefix;
        if (parts.length === 2) {
          subPrefix = parts[1].toUpperCase();
        } else if (parts.length > 2) {
          subPrefix = parts[1].toUpperCase();
        } else {
          subPrefix = 'OTHER';
        }
        
        if (!groups[mainPrefix]) {
          groups[mainPrefix] = {};
        }
        if (!groups[mainPrefix][subPrefix]) {
          groups[mainPrefix][subPrefix] = {};
        }
        groups[mainPrefix][subPrefix][key] = model;
      });
      return groups;
    },

    toggleGroup(prefix, subPrefix = null) {
      const key = subPrefix ? `${prefix}-${subPrefix}` : prefix;
      this.expandedGroups[key] = !this.expandedGroups[key];
    },

    isGroupExpanded(prefix, subPrefix = null) {
      const key = subPrefix ? `${prefix}-${subPrefix}` : prefix;
      return this.expandedGroups[key] || false;
    },
  }));
});

const { markedHighlight } = globalThis.markedHighlight;
marked.use(markedHighlight({
  langPrefix: "hljs language-",
  highlight(code, lang, _info) {
    const language = hljs.getLanguage(lang) ? lang : "plaintext";
    return hljs.highlight(code, { language }).value;
  },
}));

// **** eventsource-parser ****
class EventSourceParserStream extends TransformStream {
  constructor() {
    let parser;

    super({
      start(controller) {
        parser = createParser((event) => {
          if (event.type === "event") {
            controller.enqueue(event);
          }
        });
      },

      transform(chunk) {
        parser.feed(chunk);
      },
    });
  }
}

function createParser(onParse) {
  let isFirstChunk;
  let buffer;
  let startingPosition;
  let startingFieldLength;
  let eventId;
  let eventName;
  let data;
  reset();
  return {
    feed,
    reset,
  };
  function reset() {
    isFirstChunk = true;
    buffer = "";
    startingPosition = 0;
    startingFieldLength = -1;
    eventId = void 0;
    eventName = void 0;
    data = "";
  }
  function feed(chunk) {
    buffer = buffer ? buffer + chunk : chunk;
    if (isFirstChunk && hasBom(buffer)) {
      buffer = buffer.slice(BOM.length);
    }
    isFirstChunk = false;
    const length = buffer.length;
    let position = 0;
    let discardTrailingNewline = false;
    while (position < length) {
      if (discardTrailingNewline) {
        if (buffer[position] === "\n") {
          ++position;
        }
        discardTrailingNewline = false;
      }
      let lineLength = -1;
      let fieldLength = startingFieldLength;
      let character;
      for (
        let index = startingPosition;
        lineLength < 0 && index < length;
        ++index
      ) {
        character = buffer[index];
        if (character === ":" && fieldLength < 0) {
          fieldLength = index - position;
        } else if (character === "\r") {
          discardTrailingNewline = true;
          lineLength = index - position;
        } else if (character === "\n") {
          lineLength = index - position;
        }
      }
      if (lineLength < 0) {
        startingPosition = length - position;
        startingFieldLength = fieldLength;
        break;
      } else {
        startingPosition = 0;
        startingFieldLength = -1;
      }
      parseEventStreamLine(buffer, position, fieldLength, lineLength);
      position += lineLength + 1;
    }
    if (position === length) {
      buffer = "";
    } else if (position > 0) {
      buffer = buffer.slice(position);
    }
  }
  function parseEventStreamLine(lineBuffer, index, fieldLength, lineLength) {
    if (lineLength === 0) {
      if (data.length > 0) {
        onParse({
          type: "event",
          id: eventId,
          event: eventName || void 0,
          data: data.slice(0, -1),
          // remove trailing newline
        });

        data = "";
        eventId = void 0;
      }
      eventName = void 0;
      return;
    }
    const noValue = fieldLength < 0;
    const field = lineBuffer.slice(
      index,
      index + (noValue ? lineLength : fieldLength),
    );
    let step = 0;
    if (noValue) {
      step = lineLength;
    } else if (lineBuffer[index + fieldLength + 1] === " ") {
      step = fieldLength + 2;
    } else {
      step = fieldLength + 1;
    }
    const position = index + step;
    const valueLength = lineLength - step;
    const value = lineBuffer.slice(position, position + valueLength).toString();
    if (field === "data") {
      data += value ? "".concat(value, "\n") : "\n";
    } else if (field === "event") {
      eventName = value;
    } else if (field === "id" && !value.includes("\0")) {
      eventId = value;
    } else if (field === "retry") {
      const retry = parseInt(value, 10);
      if (!Number.isNaN(retry)) {
        onParse({
          type: "reconnect-interval",
          value: retry,
        });
      }
    }
  }
}

const BOM = [239, 187, 191];
function hasBom(buffer) {
  return BOM.every((charCode, index) => buffer.charCodeAt(index) === charCode);
}
