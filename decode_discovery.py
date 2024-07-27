import struct

def decode_mikrotik_discovery(data):
    result = {}
    # Skip the first 4 bytes
    index = 4
    #print(f"Total data length: {len(data)}")
    while index < len(data):
        #print(f"Current index: {index}")
        if index + 4 > len(data):
            #print("Not enough data for header")
            break
        type_field, length = struct.unpack('!HH', data[index:index+4])
        #print(f"Type: {type_field:02x}, Length: {length}")
        index += 4
        if index + length > len(data):
            #print(f"Not enough data for value (need {length}, have {len(data) - index})")
            break
        value = data[index:index+length]
        try:
            if type_field in [0x05, 0x07, 0x08, 0x0b, 0x0c, 0x10]:
                result[type_field] = value.decode('ascii', errors='ignore')
            elif type_field in [0x00, 0x0a, 0x0e]:
                result[type_field] = struct.unpack('!I', value.rjust(4, b'\x00'))[0]
            elif type_field == 0x11:
                result[type_field] = '.'.join(map(str, value))
            else:
                result[type_field] = value.hex()
        except Exception as e:
            print(f"Error processing field {type_field:02x}: {e}")
            result[type_field] = value.hex()
        index += length
    return result

# Your byte string
data = b'\xbf\xa2\x04\x00\x00\x01\x00\x06,\xc8\x1b\x13\x98\xe0\x00\x05\x00\x08MikroTik\x00\x07\x00\x0f6.49.1 (stable)\x00\x08\x00\x08MikroTik\x00\n\x00\x04\xd7|\xf3\x00\x00\x0b\x00\tX4N0-HRA8\x00\x0c\x00\x08RB760iGS\x00\x0e\x00\x01\x00\x00\x10\x00\rbridge/ether5\x00\x11\x00\x04\xc0\xa8X\x01'

decoded = decode_mikrotik_discovery(data)

# Print in a more readable format
for key, value in decoded.items():
    print(f"Field {key:02x}: {value}")
