#!/bin/bash
# Verify project structure is correct

PROJECT_ROOT="/nfs/shared/projects/image"

echo "=== Project Structure Verification ==="
echo ""

tree -L 3 -d "$PROJECT_ROOT" 2>/dev/null || find "$PROJECT_ROOT" -type d | sort

echo ""
echo "=== Dataset Status ==="

if [ -f "$PROJECT_ROOT/datasets/coco/train2017.zip" ]; then
    SIZE=$(du -h "$PROJECT_ROOT/datasets/coco/train2017.zip" | cut -f1)
    echo "✓ COCO dataset zip found: $SIZE"
else
    echo "✗ COCO dataset zip not found"
fi

if [ -d "$PROJECT_ROOT/datasets/coco/train2017" ]; then
    COUNT=$(find "$PROJECT_ROOT/datasets/coco/train2017" -name "*.jpg" 2>/dev/null | wc -l)
    echo "✓ Extracted images: $COUNT"
else
    echo "○ Dataset not yet extracted"
fi

echo ""
echo "=== NFS Mount Status ==="
df -h "$PROJECT_ROOT" | tail -1

echo ""
echo "=== Directory Permissions ==="
ls -la "$PROJECT_ROOT" | head -5
