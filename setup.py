from setuptools import find_packages, setup

setup(
    name="audio-qformer",
    version="0.1.0",
    packages=find_packages(),
    description="Lightweight Q-Former finetuning wrapper for Seamless M4T",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    url="https://github.com/ivanj-0/audio-qformer",
    license="MIT",
    install_requires=[],
    entry_points={
        "console_scripts": [
            "audio-qformer-train=train_qformer_llm:main",
        ],
    },
    include_package_data=True,
)
