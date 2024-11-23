API_ENDPOINT="http://${API_ENDPOINT:-$(ifconfig | grep 'inet ' | grep -v '127.0.0.1' | awk '{print $2}' | head -n 1):52415}"
echo "Using API_ENDPOINT=${API_ENDPOINT}"
docker run -d -p 3000:8080 -e OPENAI_API_BASE_URL="${API_ENDPOINT}" -e OPENAI_API_KEY=your_secret_key -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
