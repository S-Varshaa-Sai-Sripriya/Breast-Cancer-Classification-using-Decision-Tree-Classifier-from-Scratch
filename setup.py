from setuptools import setup, find_packages

setup(
    name="decision_tree_classifier",
    version="0.1.0",
    author="S Varshaa Sai Sripriya",
    description="Decision Tree Classifier on Breast Cancer Wisconsin dataset (from scratch + modular pipeline).",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "joblib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
