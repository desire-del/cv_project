from setuptools import setup, find_packages

setup(
    name="cv_project",
    version="0.1.0",
    author="desire-del | hathadd",
    description="Projet de détection et tracking de joueurs avec YOLOv8 et Deep SORT",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/desire-del/cv_project",

    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        "ipykernel",
        "numpy",
        "opencv_contrib_python",
        "opencv-python-headless",
        "pandas",
        "Pillow",
        "torch",
        "transformers",
        "ultralytics",
        "deep_sort_realtime",
        "supervision",
        "streamlit"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
