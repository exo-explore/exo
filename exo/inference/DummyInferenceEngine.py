import random  # Import missing random module
import asyncio
import numpy as np  # Import numpy for generating random outputs

class DummyInferenceEngine:
    def __init__(self, output_type="static", output_value=None, output_shape=(1,), latency_mean=0.1, latency_stddev=0.01):
        self.output_type = output_type
        self.output_value = output_value
        self.output_shape = output_shape
        self.latency_mean = latency_mean
        self.latency_stddev = latency_stddev
        
        # Validation for static output type
        if self.output_type == "static" and self.output_value is None:
            raise ValueError("output_value must be provided when output_type is 'static'.")

    async def run_inference(self):
        # Simulate latency
        latency = max(0, random.normalvariate(self.latency_mean, self.latency_stddev))  # Non-negative latency
        await asyncio.sleep(latency)
        
        # Generate output based on the specified output type
        if self.output_type == "static":
            return self.output_value  # Return the static output
        elif self.output_type == "random":
            self.output_value = np.random.randn(*self.output_shape).tolist()  # Generate random output and store it
            return self.output_value

    async def get_latency(self):
        # Simulate and return the latency
        return max(0, random.normalvariate(self.latency_mean, self.latency_stddev))


# Example usage and testing
async def test_dummy_engine():
    dummy_engine = DummyInferenceEngine(output_type="random", output_shape=(2, 2), latency_mean=0.5, latency_stddev=0.1)
    
    # Simulate inference
    output = await dummy_engine.run_inference()
    latency = await dummy_engine.get_latency()
    
    assert isinstance(output, list), "Output should be a list."
    assert isinstance(latency, float), "Latency should be a float."
    
    print(f"Simulated Inference Output: {output}")
    print(f"Simulated Inference Latency: {latency}s")

# Run the test
# asyncio.run(test_dummy_engine())
