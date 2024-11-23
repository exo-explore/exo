import SwiftUI
import WhisperKit
import AVFoundation
import Foundation
import Combine
import Vision
import AVFAudio

actor CameraActor {
    let captureSession = AVCaptureSession()
    private let photoOutput = AVCapturePhotoOutput()
    private var isConfigured = false
    private var currentPhotoCaptureDelegate: PhotoCaptureDelegate?

    func configure() throws {
        guard !isConfigured else {
            print("Camera already configured")
            return
        }

        print("Starting camera configuration")

        guard let camera = AVCaptureDevice.default(for: .video) else {
            print("No camera device available")
            throw CameraError.cameraUnavailable
        }

        do {
            let input = try AVCaptureDeviceInput(device: camera)
            print("Camera input created successfully")

            guard captureSession.canAddInput(input) else {
                print("Cannot add camera input to session")
                throw CameraError.cannotAddInputOutput
            }

            guard captureSession.canAddOutput(photoOutput) else {
                print("Cannot add photo output to session")
                throw CameraError.cannotAddInputOutput
            }

            captureSession.beginConfiguration()
            captureSession.addInput(input)
            captureSession.addOutput(photoOutput)
            captureSession.commitConfiguration()

            print("Camera session configured successfully")

            Task.detached { [weak self] in
                self?.captureSession.startRunning()
                print("Camera session started running")
            }

            isConfigured = true
            print("Camera fully configured and ready")
        } catch {
            print("Error during camera configuration: \(error)")
            throw error
        }
    }

    func capturePhoto() async throws -> String {
        guard isConfigured else {
            throw CameraError.notConfigured
        }

        return try await withCheckedThrowingContinuation { continuation in
            let photoSettings = AVCapturePhotoSettings()

            let delegate = PhotoCaptureDelegate { result in
                self.currentPhotoCaptureDelegate = nil
                continuation.resume(with: result)
            }

            self.currentPhotoCaptureDelegate = delegate

            Task { @MainActor in
                self.photoOutput.capturePhoto(with: photoSettings, delegate: delegate)
            }
        }
    }
}

class PhotoCaptureDelegate: NSObject, AVCapturePhotoCaptureDelegate {
    private let completionHandler: (Result<String, Error>) -> Void

    init(completionHandler: @escaping (Result<String, Error>) -> Void) {
        self.completionHandler = completionHandler
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error = error {
            completionHandler(.failure(error))
            return
        }

        guard let imageData = photo.fileDataRepresentation() else {
            completionHandler(.failure(CameraError.imageProcessingFailed))
            return
        }

        let base64String = imageData.base64EncodedString()
        completionHandler(.success(base64String))
    }
}

enum CameraError: Error {
    case cameraUnavailable
    case cannotAddInputOutput
    case notConfigured
    case imageProcessingFailed
}

struct CameraPreview: UIViewControllerRepresentable {
    let cameraActor: CameraActor

    func makeUIViewController(context: Context) -> UIViewController {
        let viewController = UIViewController()
        let previewLayer = AVCaptureVideoPreviewLayer(session: cameraActor.captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        viewController.view.layer.addSublayer(previewLayer)
        previewLayer.frame = viewController.view.bounds
        return viewController
    }

    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
        if let previewLayer = uiViewController.view.layer.sublayers?.first as? AVCaptureVideoPreviewLayer {
            previewLayer.frame = uiViewController.view.bounds
        }
    }
}

struct ContentView: View {
    @State private var whisperKit: WhisperKit?
    @State private var isListening = false
    @State private var currentText = ""
    @State private var bufferSeconds: Double = 0.5 // or whatever the actual buffer size is
    @State private var modelState: ModelState = .unloaded

    @AppStorage("selectedModel") private var selectedModel: String = "large-v3"
    @AppStorage("selectedLanguage") private var selectedLanguage: String = "english"
    @AppStorage("selectedTask") private var selectedTask: String = "transcribe"

    @State private var isRecordingMemo = false
    @State private var currentMemo = ""
    @State private var lastVoiceActivityTime = Date()
    @State private var silenceTimer: Timer?
    @State private var voiceActivityThreshold: Float = 0.40
    @State private var silenceTimeThreshold = 1.0
    @State private var debugText = ""
    @State private var apiEndpoint = "http://192.168.212.74:52415/v1/chat/completions"
    @State private var audioBuffer: [Float] = []
    @State private var bufferDuration: Double = 0.5 // 0.5 seconds buffer
    @State private var isInitialTranscription = true
    @State private var streamingResponse = ""
    @State private var cancellables = Set<AnyCancellable>()

    @State private var cameraActor: CameraActor?
    @State private var showLiveCamera = false
    @State private var capturedImageBase64: String?
    @State private var errorMessage: String?
    @State private var isCameraReady = false

    @State private var speechSynthesizer = AVSpeechSynthesizer()
    @State private var speechBuffer = ""
    @State private var wordCount = 0
    let maxWords = 12
    @State private var originalSilenceThreshold: Float = 0.40
    @State private var isTTSActive: Bool = false
    @State private var canRecordAudio: Bool = true
    @State private var ttsFinishTime: Date?

    @State private var isRequestInProgress = false
    @State private var isFirst3WordsOfResponse = true

    var body: some View {
        ZStack {
            if showLiveCamera, isCameraReady, let actor = cameraActor {
                CameraPreview(cameraActor: actor)
                    .edgesIgnoringSafeArea(.all)
            }

            ScrollView {
                VStack {
                    Text(currentText)
                        .padding()

                    Text(isListening ? "Listening..." : "Not listening")
                        .foregroundColor(isListening ? .green : .red)

                    if isRecordingMemo {
                        Text("Recording...")
                            .foregroundColor(.blue)
                    }

                    Picker("Model", selection: $selectedModel) {
                        Text("large-v3").tag("large-v3")
                        Text("base").tag("base")
                        Text("small").tag("small")
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    .padding()

                    Button("Load Model") {
                        loadModel(selectedModel)
                    }
                    .disabled(modelState == .loaded)
                    .padding()

                    Text("Model State: \(modelState.description)")

                    Text(debugText)
                        .font(.caption)
                        .foregroundColor(.gray)

                    Text("TTS Active: \(isTTSActive ? "Yes" : "No")")
                        .font(.caption)
                        .foregroundColor(isTTSActive ? .green : .red)

                    Text("Current Silence Threshold: \(voiceActivityThreshold, specifier: "%.2f")")
                        .font(.caption)
                        .foregroundColor(.blue)

                    Text("Original Silence Threshold: \(originalSilenceThreshold, specifier: "%.2f")")
                        .font(.caption)
                        .foregroundColor(.orange)

                    Slider(value: $voiceActivityThreshold, in: 0.01...1.0) {
                        Text("Voice Activity Threshold: \(voiceActivityThreshold, specifier: "%.2f")")
                    }

                    Text("API Response:")
                        .font(.headline)
                        .padding(.top)

                    ScrollView {
                        Text(streamingResponse)
                            .padding()
                    }
                    .frame(height: 200)
                    .border(Color.gray, width: 1)

                    Toggle("Show Live Camera", isOn: $showLiveCamera)
                        .padding()
                        .onChange(of: showLiveCamera) { newValue in
                            if newValue {
                                Task {
                                    await setupCamera()
                                }
                            } else {
                                cameraActor = nil
                                isCameraReady = false
                                print("Camera disabled")
                            }
                        }

                    if !showLiveCamera {
                        Text("Camera Ready: \(isCameraReady ? "Yes" : "No")")
                            .padding()

                        if let errorMessage = errorMessage {
                            Text("Error: \(errorMessage)")
                                .foregroundColor(.red)
                                .padding()
                        }
                    }
                }
            }
            .opacity(showLiveCamera ? 0.7 : 1)
        }
        .onAppear {
            setupWhisperKit()
            startTTSMonitoring()
        }
    }

    private func setupWhisperKit() {
        Task {
            do {
                whisperKit = try await WhisperKit(verbose: true)
                print("WhisperKit initialized successfully")
                startListening()
                startAudioBuffering()
            } catch {
                print("Error initializing WhisperKit: \(error)")
            }
        }
    }

    private func startTTSMonitoring() {
        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            let newTTSActive = speechSynthesizer.isSpeaking
            if newTTSActive != isTTSActive {
                isTTSActive = newTTSActive
                canRecordAudio = !newTTSActive
                if isTTSActive {
                    voiceActivityThreshold = 1.0 // Set to max to prevent recording
                    whisperKit?.audioProcessor.purgeAudioSamples(keepingLast: 0) // Flush audio buffer
                    print("TTS Started - Audio recording paused")
                } else {
                    ttsFinishTime = Date()
                    print("TTS Finished - Waiting 0.5 seconds before resuming audio recording")
                }
                updateDebugText()
            }

            if !isTTSActive, let finishTime = ttsFinishTime, Date().timeIntervalSince(finishTime) >= 0.5 {
                whisperKit?.audioProcessor.purgeAudioSamples(keepingLast: 0) // Flush audio buffer
                voiceActivityThreshold = originalSilenceThreshold
                canRecordAudio = true
                ttsFinishTime = nil
                print("Audio recording resumed after TTS delay")
                updateDebugText()
            }
        }
    }

    private func updateDebugText() {
        debugText += "\nTTS Active: \(isTTSActive)"
        debugText += "\nCurrent Silence Threshold: \(voiceActivityThreshold)"
        debugText += "\nOriginal Silence Threshold: \(originalSilenceThreshold)"
        debugText += "\n---"
    }

    private func startAudioBuffering() {
        Task {
            while true {
                if let samples = whisperKit?.audioProcessor.audioSamples {
                    let bufferSize = Int(Double(WhisperKit.sampleRate) * bufferDuration)
                    audioBuffer = Array(samples.suffix(bufferSize))
                }
                try await Task.sleep(nanoseconds: 100_000_000) // Update every 0.1 seconds
            }
        }
    }

    private func loadModel(_ model: String) {
        Task {
            let success = try await loadModel(selectedModel)
            if success {
                startListening()
            } else {
                print("Model failed to load, cannot start listening")
            }
        }
    }

    private func startListening() {
        guard let audioProcessor = whisperKit?.audioProcessor else {
            print("AudioProcessor not available")
            return
        }

        do {
            try audioProcessor.startRecordingLive { buffer in
                DispatchQueue.main.async {
                    checkVoiceActivity()
                }
            }
            isListening = true
        } catch {
            print("Error starting listening: \(error)")
        }
    }

    private func checkVoiceActivity() {
        guard canRecordAudio, let audioProcessor = whisperKit?.audioProcessor else { return }

        let voiceDetected = AudioProcessor.isVoiceDetected(
            in: audioProcessor.relativeEnergy,
            nextBufferInSeconds: Float(bufferSeconds),
            silenceThreshold: Float(voiceActivityThreshold)
        )

        let energyValuesToConsider = Int(Float(bufferSeconds) / 0.1)
        let nextBufferEnergies = audioProcessor.relativeEnergy.suffix(energyValuesToConsider)
        let numberOfValuesToCheck = max(10, nextBufferEnergies.count - 10)
        let relevantEnergies = Array(nextBufferEnergies.prefix(numberOfValuesToCheck))

        debugText = """
        Buffer seconds: \(bufferSeconds)
        Energy values to consider: \(energyValuesToConsider)
        Number of values to check: \(numberOfValuesToCheck)
        Silence threshold: \(voiceActivityThreshold)
        Relevant energies: \(relevantEnergies)
        Max energy: \(relevantEnergies.max() ?? 0)
        Voice detected: \(voiceDetected)
        """

        if voiceDetected {
            lastVoiceActivityTime = Date()
            if !isRecordingMemo {
                startNewMemo()
            }
        } else {
            checkSilence()
        }
    }

    private func checkSilence() {
        let silenceDuration = Date().timeIntervalSince(lastVoiceActivityTime)
        debugText += "\nSilence duration: \(silenceDuration)"

        if silenceDuration > silenceTimeThreshold {
            endCurrentMemo()
        }
    }

    private func endCurrentMemo() {
        if isRecordingMemo {
            isRecordingMemo = false
            silenceTimer?.invalidate()
            silenceTimer = nil
            if !currentMemo.isEmpty {
                saveMemoToFile(currentMemo)
                currentMemo = ""
            }
            currentText = ""
            whisperKit?.audioProcessor.purgeAudioSamples(keepingLast: 0)
            print("Ended memo")
            debugText += "\nMemo ended"
        }
    }

    private func startNewMemo() {
        isRecordingMemo = true
        currentMemo = ""
        isInitialTranscription = true
        silenceTimer?.invalidate()
        silenceTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
            checkSilence()
        }
        transcribeInRealTime()
        print("Started new memo")
    }

    private func transcribeInRealTime() {
        Task {
            while isRecordingMemo {
                if canRecordAudio, let samples = whisperKit?.audioProcessor.audioSamples, samples.count > WhisperKit.sampleRate {
                    do {
                        let samplesToTranscribe: [Float]
                        if isInitialTranscription {
                            samplesToTranscribe = audioBuffer + samples
                            isInitialTranscription = false
                        } else {
                            samplesToTranscribe = Array(samples)
                        }

                        let result = try await whisperKit?.transcribe(audioArray: samplesToTranscribe)
                        await MainActor.run {
                            let newText = result?.first?.text ?? ""
                            if !newText.isEmpty {
                                currentMemo = newText
                                currentText = newText
                            }
                        }
                    } catch {
                        print("Transcription error: \(error)")
                    }
                }
                try await Task.sleep(nanoseconds: 500_000_000) // Sleep for 0.5 seconds
            }
        }
    }

    private func saveMemoToFile(_ memo: String) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let fileName = "memo_\(dateFormatter.string(from: Date())).txt"

        guard let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            print("Unable to access documents directory")
            return
        }

        let fileURL = documentsDirectory.appendingPathComponent(fileName)

        do {
            try memo.write(to: fileURL, atomically: true, encoding: .utf8)
            print("Memo saved to: \(fileURL.path)")
        } catch {
            print("Error saving memo: \(error)")
        }

        Task {
            if !isCameraReady {
                print("Camera not ready, initializing...")
                await setupCamera()
            }

            if let imageBase64 = await capturePhotoBase64() {
                sendMemoToAPI(memo, imageBase64: imageBase64)
            } else {
                sendMemoToAPI(memo)
            }
        }
    }

    private func setupCamera() async {
        print("Setting up camera...")
        do {
            let actor = CameraActor()
            print("CameraActor instance created")
            try await actor.configure()
            print("Camera configured successfully")
            await MainActor.run {
                self.cameraActor = actor
                self.errorMessage = nil
                self.isCameraReady = true
                print("Camera setup complete, UI updated")
            }
        } catch {
            print("Camera setup failed: \(error)")
            await MainActor.run {
                self.errorMessage = "Failed to initialize camera: \(error.localizedDescription)"
                self.isCameraReady = false
                print("Camera setup failure reflected in UI")
            }
        }
    }

    private func capturePhotoBase64() async -> String? {
        print("Attempting to capture photo...")
        if !isCameraReady {
            print("Camera not ready, attempting to initialize...")
            await setupCamera()
        }

        guard let actor = cameraActor, isCameraReady else {
            print("Camera not initialized or not ready, cannot capture photo")
            await MainActor.run {
                self.errorMessage = "Camera not initialized or not ready"
            }
            return nil
        }

        do {
            let base64String = try await actor.capturePhoto()
            print("Photo captured successfully")
            await MainActor.run {
                self.errorMessage = nil
            }
            return base64String
        } catch {
            print("Error capturing photo: \(error)")
            await MainActor.run {
                self.errorMessage = "Failed to capture photo: \(error.localizedDescription)"
            }
            return nil
        }
    }

    private func sendMemoToAPI(_ memo: String, imageBase64: String? = nil) {
        Task {
            guard !isRequestInProgress else {
                print("A request is already in progress. Skipping this one.")
                return
            }

            isRequestInProgress = true
            isFirst3WordsOfResponse = true  // Reset for new request
            defer { isRequestInProgress = false }

            do {
                print("Starting API request for memo: \(memo.prefix(50))...")

                guard let url = URL(string: apiEndpoint) else {
                    print("Invalid API endpoint URL: \(apiEndpoint)")
                    return
                }

                var payload: [String: Any] = [
                    "model": "llava-1.5-7b-hf",
                    "messages": [
                        ["role": "user", "content": [
                            ["type": "text", "text": "You are a helpful conversational assistant chatting with a Gen Z user using their iPhone for voice transcription and sending images to you with their iPhone camera. Be conversational and concise, with a laid back attitude and be cheerful with humour. User said: " + memo],
                        ]]
                    ],
                    "temperature": 0.7,
                    "stream": true
                ]

                if let imageBase64 = imageBase64 {
                    if var userMessage = (payload["messages"] as? [[String: Any]])?.last,
                       var content = userMessage["content"] as? [[String: Any]] {
                        content.append(["type": "image_url", "image_url": ["url": "data:image/jpeg;base64,\(imageBase64)"]])
                        userMessage["content"] = content
                        payload["messages"] = [userMessage]
                    }
                }

                guard let jsonData = try? JSONSerialization.data(withJSONObject: payload) else {
                    print("Failed to serialize JSON payload")
                    return
                }

                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = jsonData

                print("Sending request to \(url.absoluteString)")

                await MainActor.run {
                    self.streamingResponse = ""
                }

                let (bytes, response) = try await URLSession.shared.bytes(for: request)

                guard let httpResponse = response as? HTTPURLResponse else {
                    print("Invalid response")
                    return
                }

                print("Response status code: \(httpResponse.statusCode)")

                for try await line in bytes.lines {
                    print("Received line: \(line)")
                    await processStreamLine(line)
                }

                print("Stream completed")
            } catch {
                print("Error: \(error.localizedDescription)")
            }
        }
    }

    private func processStreamLine(_ line: String) async {
        let jsonString: String
        if line.hasPrefix("data: ") {
            jsonString = String(line.dropFirst(6))
        } else {
            jsonString = line
        }

        if jsonString.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return
        }

        if let jsonData = jsonString.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: jsonData, options: []) as? [String: Any],
           let choices = json["choices"] as? [[String: Any]],
           let firstChoice = choices.first,
           let delta = firstChoice["delta"] as? [String: String],
           let content = delta["content"] {
            print("Extracted content: \(content)")
            await MainActor.run {
                self.streamingResponse += content
                bufferContent(content)
            }
        }
    }

    private func bufferContent(_ content: String) {
        speechBuffer += content
        let words = speechBuffer.split(separator: " ")
        wordCount = words.count

        if isFirst3WordsOfResponse && wordCount >= 3 {
            isFirst3WordsOfResponse = false
            speakBufferedContent()
        } else if content.contains(".") || content.contains("!") || content.contains("?") || wordCount >= maxWords {
            speakBufferedContent()
        }
    }

    private func speakBufferedContent() {
        guard !speechBuffer.isEmpty else { return }
        speakContent(speechBuffer)
        speechBuffer = ""
        wordCount = 0
    }

    private func speakContent(_ content: String) {
        let utterance = AVSpeechUtterance(string: content)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.5
        speechSynthesizer.speak(utterance)
    }

    private func loadModel(_ model: String) async throws -> Bool {
        guard let whisperKit = whisperKit else {
            print("WhisperKit instance not initialized")
            return false
        }
        modelState = .loading
        do {
            print("Starting to load model: \(model)")
            try await whisperKit.loadModels()
            await MainActor.run {
                modelState = .loaded
                print("Model loaded successfully: \(model)")
            }
            return true
        } catch {
            print("Error loading model: \(error)")
            await MainActor.run { modelState = .unloaded }
            return false
        }
    }

    private func capturePhoto() async {
        print("Attempting to capture photo...")
        print("Camera ready: \(isCameraReady), CameraActor exists: \(cameraActor != nil)")
        guard let actor = cameraActor, isCameraReady else {
            print("Camera not initialized or not ready, cannot capture photo")
            await MainActor.run {
                self.errorMessage = "Camera not initialized or not ready"
            }
            return
        }

        do {
            let base64String = try await actor.capturePhoto()
            print("Photo captured successfully")
            await MainActor.run {
                self.capturedImageBase64 = base64String
                self.errorMessage = nil
            }
        } catch {
            print("Error capturing photo: \(error)")
            await MainActor.run {
                self.errorMessage = "Failed to capture photo: \(error.localizedDescription)"
            }
        }
    }
}
