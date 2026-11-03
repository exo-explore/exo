"""What a request is told when it ends with an error, where several parts of exo end requests
the same way."""

# A request sent to an instance that was deleted (the model was stopped, or a node was lost)
# just before the request could start
INSTANCE_GONE_BEFORE_START = (
    "The model instance was shut down before this request could start."
)
