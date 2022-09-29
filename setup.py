import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
  long_description = fh.read()

setuptools.setup(
  name="baseline",
  version="0.0.3",
  author="Yiheng Liu and Enjie Ge",
  description="A package to help compare functional brain network's performance.",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/SNNUBIAI/Baseline",
  packages=setuptools.find_packages(),
  package_data={'baseline.templates.data': ["*.npy", "*/*.mat"]},
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
)