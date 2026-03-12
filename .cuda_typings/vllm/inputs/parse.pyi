from .data import ProcessorInputs as ProcessorInputs, SingletonInputs as SingletonInputs

def split_enc_dec_inputs(
    inputs: ProcessorInputs,
) -> tuple[SingletonInputs | None, SingletonInputs]: ...
