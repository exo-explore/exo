import logging
from typing import List
from exo.topology.partitioning_strategy import PartitioningStrategy, Partition
from exo.topology.topology import Topology

class LatencyAwareMemoryAndFlopsPartitioningStrategy(PartitioningStrategy):
    
    def partition(self, topology: Topology) -> List[Partition]:
        logging.info("Starting partition method")
        
        # Extract nodes and their capabilities
        nodes = list(topology.all_nodes())
        logging.debug(f"Nodes: {nodes}")
        
        if not nodes:
            logging.warning("No nodes found in topology")
            return []
        
        # Normalize memory and FLOPS
        max_memory = max(node[1].memory for node in nodes)
        max_flops = max(node[1].flops.fp32 for node in nodes)
        logging.debug(f"Max memory: {max_memory}, Max FLOPS: {max_flops}")
        
        # Normalize latency and throughput
        max_latency = max(latency for _, neighbors in topology.peer_graph.items() 
                          for _, (latency, _) in neighbors.items()) if topology.peer_graph else 1
        max_throughput = max(throughput for _, neighbors in topology.peer_graph.items() 
                             for _, (_, throughput) in neighbors.items()) if topology.peer_graph else 1
        logging.debug(f"Max latency: {max_latency}, Max throughput: {max_throughput}")
        
        # Sort nodes based on the combined metric in descending order
        nodes.sort(key=lambda node: self.combined_metric(node, topology, max_memory, max_flops, max_latency, max_throughput), reverse=True)
        logging.debug(f"Sorted nodes: {nodes}")
        
        # Calculate total combined metric
        total_combined_metric = sum(self.combined_metric(node, topology, max_memory, max_flops, max_latency, max_throughput) for node in nodes)
        logging.debug(f"Total combined metric: {total_combined_metric}")
        
        # Initialize partitions
        partitions = []
        start = 0
        
        # Create partitions based on the combined metric
        for node in nodes:
            end = round(start + (self.combined_metric(node, topology, max_memory, max_flops, max_latency, max_throughput) / total_combined_metric), 5)
            partitions.append(Partition(node[0], start, end))
            logging.debug(f"Partition created for node: {node[0]}, Start: {start}, End: {end}")
            start = end
        
        logging.info("Partition method completed")
        return partitions
    
    def combined_metric(self, node, topology, max_memory, max_flops, max_latency, max_throughput):
        # Normalize memory and FLOPS
        normalized_memory = node[1].memory / max_memory
        normalized_flops = node[1].flops.fp32 / max_flops
        logging.debug(f"Node: {node[0]}, Normalized memory: {normalized_memory}, Normalized FLOPS: {normalized_flops}")
        
        # Sum latencies and throughputs to all other nodes
        neighbors = topology.get_neighbors(node[0])
        total_latency = sum(latency for _, (latency, _) in neighbors.items()) if neighbors else 0
        total_throughput = sum(throughput for _, (_, throughput) in neighbors.items()) if neighbors else 0
        
        # Normalize total latency and throughput
        normalized_total_latency = total_latency / (max_latency * len(neighbors)) if max_latency != 0 else 0
        normalized_total_throughput = total_throughput / (max_throughput * len(neighbors)) if max_throughput != 0 else 0
        
        logging.debug(f"Node: {node[0]}, Normalized total latency: {normalized_total_latency}, Normalized total throughput: {normalized_total_throughput}")
        
        # Combine normalized values with weights
        # Example: Higher memory, higher FLOPS, lower total latency, higher total throughput are desirable
        # Adjust the formula to reflect this priority
        epsilon = 1e-9  # Small epsilon value to avoid division by zero
        
        combined = (
            normalized_memory +
            normalized_flops +
            1 / (normalized_total_latency + epsilon) +
            normalized_total_throughput
        )
        logging.debug(f"Node: {node[0]}, Combined metric: {combined}")
        return combined