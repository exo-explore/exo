document.addEventListener("alpine:init", () => {
  Alpine.data("state", () => ({
    // current state
    cstate: {
      time: null,
      messages: [],
      selectedModel: 'llama-3.1-8b',
    },

    // historical state
    histories: JSON.parse(localStorage.getItem("histories")) || [],

    home: 0,
    generating: false,
    endpoint: `${window.location.origin}/v1`,
    errorMessage: null,

    // performance tracking
    time_till_first: 0,
    tokens_per_second: 0,
    total_tokens: 0,

    // image handling
    imagePreview: null,

    removeHistory(cstate) {
      const index = this.histories.findIndex((state) => {
        return state.time === cstate.time;
      });
      if (index !== -1) {
        this.histories.splice(index, 1);
        localStorage.setItem("histories", JSON.stringify(this.histories));
      }
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
        const imageInput = document.getElementById("image-upload");

        // Check if there's an image file selected
        const hasImage = imageInput && imageInput.files && imageInput.files.length > 0;
        
        if (!value && !this.imagePreview) return;

        if (this.generating) return;
        this.generating = true;
        if (this.home === 0) this.home = 1;

        // ensure that going back in history will go back to home
        window.history.pushState({}, "", "/");

        // add message to list
        if (!hasImage && value) {
          this.cstate.messages.push({ role: "user", content: value });
        }

        // clear textarea
        el.value = "";
        el.style.height = "auto";
        el.style.height = el.scrollHeight + "px";

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


        // start receiving server sent events
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
        console.error('error', error)
        this.errorMessage = error;
        setTimeout(() => {
          this.errorMessage = null;
        }, 5 * 1000)
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
      // stream response
      console.log("model", model)
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
        if (done) {
          break;
        }
        if (value.type === "event") {
          const json = JSON.parse(value.data);
          if (json.choices) {
            const choice = json.choices[0];
            if (choice.finish_reason === "stop") {
              break;
            }
            yield choice.delta.content;
          }
        }
      }
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
