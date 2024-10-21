import yaml

with open('topology.yml', 'r') as f:
    edgenodes = yaml.safe_load(f)

for node in edgenodes:
    print(f"{node['server']} {node['id']}:")
    print(f"  Adresse: {node['address']} {node['port']}")
    print(f"  Capabilities:")
    for capability, value in node['device_capabilities'].items():
        print(f"    {capability}: {value}")
        if f"{capability}" == "flops":
            for flopstr, flopvalue in node['device_capabilities']['flops'].items():
                print(f"    {flopstr}: {flopvalue}")

