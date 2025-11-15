#!/usr/bin/env python3
"""
Check if all required dependencies are installed.
"""

import sys

required_modules = [
    ('cv2', 'opencv-python'),
    ('torch', 'torch'),
    ('clip', 'git+https://github.com/openai/CLIP.git'),
    ('transformers', 'transformers'),
    ('PIL', 'Pillow'),
    ('numpy', 'numpy'),
    ('matplotlib', 'matplotlib'),
    ('moviepy', 'moviepy'),
]

missing = []

for module_name, package_name in required_modules:
    try:
        if module_name == 'PIL':
            __import__('PIL')
        elif module_name == 'clip':
            import clip
        else:
            __import__(module_name)
        print(f"✓ {package_name}")
    except ImportError:
        print(f"✗ {package_name} - MISSING")
        missing.append(package_name)

if missing:
    print(f"\nMissing packages: {', '.join(missing)}")
    print("\nInstall with:")
    print(f"  pip install {' '.join([p for p in missing if not p.startswith('git+')])}")
    if any(p.startswith('git+') for p in missing):
        print(f"  pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)
else:
    print("\nAll dependencies are installed!")
    sys.exit(0)


