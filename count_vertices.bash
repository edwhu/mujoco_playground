#!/bin/bash
folder="${1:-.}"
total=0
for file in "$folder"/*.obj; do
    if [ -f "$file" ]; then
        count=$(grep -c "^v " "$file")
        echo "$(basename "$file"): $count vertices"
        total=$((total + count))
    fi
done
echo "Total vertices: $total"