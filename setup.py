from setuptools import setup, find_packages

setup(
    name='kompetisi-telkom',  # Ganti nama proyek sesuka Anda
    version='0.1.0',
    
    # --- BAGIAN KRUSIAL UNTUK STRUKTUR SRC ---
    
    # 1. Beritahu Python bahwa root paket ada di dalam folder 'src'
    package_dir={'': 'src'},
    
    # 2. Cari semua paket secara otomatis di dalam folder 'src'
    # Ini akan otomatis menemukan 'modeling', 'utilities', 'utilities.yolov5', dll.
    packages=find_packages(where='src'),
    
    # 3. Dependencies (Opsional, biar sekalian install library yang dibutuhkan)
    install_requires=[
        'numpy',
        'torch',
        # tambahkan library lain jika perlu
    ],
    
    python_requires='>=3.8',
)