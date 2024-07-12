import setuptools

#with open("README.md", "r") as fh:
#    long_description = fh.read()

setuptools.setup(
    name="lmcat",
    version="0.1",
    author="Hao Gao",
    author_email="gaaooh@126.com",
    description="Python codes for analysis of iquid metal catalysts.",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    #url="https://github.com/gooaah/pycqg",
    # include_package_data=True,
    # exclude_package_date={'':['.gitignore']},
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
