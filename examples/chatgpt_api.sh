# exo provides an API that aims to be a drop-in replacements for the ChatGPT-API.
# This example shows how you can use the API first without streaming and second with streaming.
# This works the same in a single-node set up and in a multi-node setup.
# You need to start exo before running this by running `python3 main.py`.

API_ENDPOINT="http://${API_ENDPOINT:-$(ifconfig | grep 'inet ' | grep -v '127.0.0.1' | awk '{print $2}' | head -n 1):52415}"
MODEL="llama-3.1-8b"
PROMPT="What is the meaning of exo?"
TEMPERATURE=0.7

echo ""
echo ""
echo "--- Output without streaming:"
echo ""
curl "${API_ENDPOINT}/v1/chat/completions" --silent \
  -H "Content-Type: application/json" \
  -d '{
     "model": "'"${MODEL}"'",
     "messages": [{"role": "user", "content": "'"${PROMPT}"'"}],
     "temperature": '"${TEMPERATURE}"'
   }'

echo ""
echo ""
echo "--- Output with streaming:"
echo ""
curl "${API_ENDPOINT}/v1/chat/completions" --silent \
  -H "Content-Type: application/json" \
  -d '{
     "model": "'"${MODEL}"'",
     "messages": [{"role": "user", "content": "'"${PROMPT}"'"}],
     "temperature": '"${TEMPERATURE}"',
     "stream": true
   }' | while read -r line; do
       if [[ $line == data:* ]]; then
           content=$(echo "$line" | sed 's/^data: //')
           echo "$content" | jq -r '.choices[].delta.content' --unbuffered | tr -d '\n'
       fi
   done