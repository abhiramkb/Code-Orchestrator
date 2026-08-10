import argparse

parser = argparse.ArgumentParser(description="Parse arbitrary arguments.")

# parse_known_args returns (known_args_namespace, unknown_args_list)
_, unknown = parser.parse_known_args()

# Process arbitrary arguments into a dictionary
parsed_args = {}
i = 0
while i < len(unknown):
    item = unknown[i]
    if item.startswith("--") or item.startswith("-"):
        key = item.lstrip("-")
        # Check if the next item exists and is a value (not another flag)
        if i + 1 < len(unknown) and not unknown[i + 1].startswith("-"):
            parsed_args[key] = unknown[i + 1]
            i += 2
        else:
            # Standalone flag (boolean True)
            parsed_args[key] = True
            i += 1
    else:
        # Positional value without a flag name
        parsed_args[f"positional_{i}"] = item
        i += 1

# Print received key-value pairs
for key, value in parsed_args.items():
    print(f"{key}: {value}")