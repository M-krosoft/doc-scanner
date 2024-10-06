from setuptools import setup, find_packages

setup(
    name="doc_scanner",
    version="1.0.0-alpha",
    packages=find_packages(),
    install_requires=[
        "imageio>=2.35.1",
        "imutils>=0.5.4",
        "lazy_loader>=0.4",
        "networkx>=3.3",
        "numpy>=2.1.1",
        "opencv-python>=4.10.0.84",
        "packaging>=24.1",
        "pillow>=10.4.0",
        "scikit-image>=0.24.0",
        "scipy>=1.14.1",
        "setuptools>=75.1.0",
        "tifffile>=2024.9.20"
    ],
    author="Makrosoft",
    author_email="devjerzy@gmail.com",
    description="Package for scanning documents",
    url="https://github.com/M-krosoft/doc-scanner",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
)
