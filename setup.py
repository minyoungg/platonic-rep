import setuptools

setuptools.setup(
    name="platonic",
    packages=setuptools.find_packages(),
    version="0.0.1",
    author="minyoung huh",
    author_email="minhuh@mit.edu",
    description="platonic-rep",
    url="git@github.com:anon/platonic-rep.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
            "torch>=2.0.0",
    ],
    python_requires='>=3.10',
)