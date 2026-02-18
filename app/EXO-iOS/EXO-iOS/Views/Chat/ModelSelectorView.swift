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
            .scrollContentBackground(.hidden)
            .background(Color.exoBlack)
            .navigationTitle("SELECT MODEL")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                        .font(.exoSubheadline)
                        .foregroundStyle(Color.exoYellow)
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
        .foregroundStyle(Color.exoLightGray)
        .listRowBackground(Color.exoBlack)
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
            .listRowBackground(Color.exoDarkGray)
        }
    }

    private func modelRow(_ model: ModelOption) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.displayName)
                    .font(.exoSubheadline)
                    .foregroundStyle(Color.exoForeground)
                Text(model.id)
                    .font(.exoCaption)
                    .foregroundStyle(Color.exoLightGray)
            }

            Spacer()

            if model.id == selectedModelId {
                Image(systemName: "checkmark")
                    .font(.exoSubheadline)
                    .foregroundStyle(Color.exoYellow)
            }
        }
    }
}
