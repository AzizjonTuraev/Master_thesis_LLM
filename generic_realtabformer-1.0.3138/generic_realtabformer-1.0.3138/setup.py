import setuptools

with open("README.md", "r") as f:
    description = f.read()

# with open("requirements.txt", 'r') as f:
#     dependencies = f.read().splitlines()


if __name__ == "__main__":
    setuptools.setup(setup_requires=["setuptools_scm"], include_package_data=True, version="1.0.3138",
    # install_requires=dependencies,
    install_requires=[
        "datasets==2.18.0",
        "torch==2.2.0",
        "scikit-learn==1.4.1.post1",
        "transformers==4.39.0",
        "shapely>=2.0",
        "transformers[torch]",
        "accelerate==0.28.0"
    ],
    long_description=description,
    long_description_content_type="text/markdown"
    )
