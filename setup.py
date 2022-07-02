import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchquanter",            # 模块名称
    version="1.0",          # 当前版本
    author="roxbili",           # 作者
    description="quant for torch",    # 模块简介
    long_description=long_description,          # 模块详细介绍
    long_description_content_type="text/markdown",  # 模块详细介绍格式
    url="https://github.com/Roxbili/TorchQuanter",                    # 模块github地址
    packages=setuptools.find_packages(),        # 自动找到项目中导入的模块
    # 模块相关的元数据（更多的描述）
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Independent",
    ],
    # 依赖模块
    install_requires=[
        "torch"
    ],
  # python版本
    python_requires=">=3",
)