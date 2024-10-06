from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]


setup(
    name="doc-scanner",
    version="1.0.0-alpha",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    author="Makrosoft",
    author_email="devjerzy@gmail.com",
    description="Package for scanning documents",
    url="https://github.com/M-krosoft/doc-scanner",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
)
