import SwiftUI

struct ModelSelectorView: View {
    let models: [ModelOption]
    let selectedModelId: String?
    let onSelect: (String) -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                if models.isEmpty {
                    emptyContent
                } else {
                    modelsList
                }
            }
            .navigationTitle("Select Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }

    private var emptyContent: some View {
        ContentUnavailableView(
            "No Models Available",
            systemImage: "cpu",
            description: Text("Connect to an EXO cluster to see available models.")
        )
    }

    private var modelsList: some View {
        ForEach(models) { model in
            Button {
                onSelect(model.id)
                dismiss()
            } label: {
                modelRow(model)
            }
            .tint(.primary)
        }
    }

    private func modelRow(_ model: ModelOption) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.displayName)
                    .fontWeight(.medium)
                Text(model.id)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if model.id == selectedModelId {
                Image(systemName: "checkmark")
                    .foregroundStyle(Color.accentColor)
            }
        }
    }
}
