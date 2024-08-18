import SwiftUI
import WhisperKit
import AVFoundation
import Foundation

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
    @State private var voiceActivityThreshold: Float = 0.1 // Lower this value
    @State private var silenceTimeThreshold = 1.0
    @State private var debugText = ""
    @State private var apiEndpoint = "http://192.168.212.74:8000/v1/chat/completions"
    @State private var audioBuffer: [Float] = []
    @State private var bufferDuration: Double = 0.5 // 0.5 seconds buffer
    @State private var isInitialTranscription = true

    var body: some View {
        VStack {
            Text(currentText)
                .padding()

            Text(isListening ? "Listening..." : "Not listening")
                .foregroundColor(isListening ? .green : .red)

            if isRecordingMemo {
                Text("Recording memo...")
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

            Slider(value: $voiceActivityThreshold, in: 0.01...1.0) {
                Text("Voice Activity Threshold: \(voiceActivityThreshold, specifier: "%.2f")")
            }
        }
        .onAppear {
            setupWhisperKit()
        }
    }

    private func setupWhisperKit() {
        Task {
            do {
                whisperKit = try await WhisperKit(verbose: true)
                print("WhisperKit initialized successfully")
                startListening()
                startAudioBuffering() // Add this line
            } catch {
                print("Error initializing WhisperKit: \(error)")
            }
        }
    }

    // Add this new function
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
        guard let audioProcessor = whisperKit?.audioProcessor else { return }

        let voiceDetected = AudioProcessor.isVoiceDetected(
            in: audioProcessor.relativeEnergy,
            nextBufferInSeconds: Float(bufferSeconds),
            silenceThreshold: Float(voiceActivityThreshold)
        )

        // Debug logging
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
            // Flush the transcribed text and reset audio samples
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
                if let samples = whisperKit?.audioProcessor.audioSamples, samples.count > WhisperKit.sampleRate {
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

        // After saving to file, send HTTP request
        sendMemoToAPI(memo)
    }

    private func sendMemoToAPI(_ memo: String) {
        guard let url = URL(string: apiEndpoint) else {
            print("Invalid API endpoint URL")
            return
        }

        let payload: [String: Any] = [
            "model": "llama-3.1-8b",
            "messages": [["role": "user", "content": memo]],
            "temperature": 0.7,
            "stream": true
        ]

        guard let jsonData = try? JSONSerialization.data(withJSONObject: payload) else {
            print("Failed to serialize JSON payload")
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = jsonData

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("Error sending memo to API: \(error)")
                return
            }

            if let httpResponse = response as? HTTPURLResponse {
                print("API response status code: \(httpResponse.statusCode)")
            }

            if let data = data, let responseString = String(data: data, encoding: .utf8) {
                print("API response: \(responseString)")
            }
        }.resume()
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
}