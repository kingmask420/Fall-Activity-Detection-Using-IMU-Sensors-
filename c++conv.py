# convert_tflite_to_c_array.py

def convert_to_c_array(file_path, output_path):
    with open(file_path, "rb") as f:
        data = f.read()

    with open(output_path, "w") as f:
        f.write("unsigned char model_tflite[] = {\n")
        for i, byte in enumerate(data):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
        f.write("\n};\n")
        f.write(f"unsigned int model_tflite_len = {len(data)};\n")

# Usage
convert_to_c_array("model.tflite", "model_data.cc")
