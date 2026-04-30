# swap_columns.py
import os
os.chdir(os.path.dirname(__file__))
input_file = "train.txt"
output_file = "train.txt"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    left, right = line.split(",")
    new_lines.append(f"{right},{left}\n")

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("处理完成，结果已写入", output_file)
