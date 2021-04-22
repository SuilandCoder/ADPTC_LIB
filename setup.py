'''
Description: 
Author: SongJ
Date: 2020-12-28 10:16:13
LastEditTime: 2021-04-12 10:45:16
LastEditors: SongJ
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 10:16
# @Author  : SongJ
# @Site    : 
# @File    : setup.py
# @Software: VS Code
# @Description:

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "ADPTC_LIB",
    version = "0.0.7",
    author = "SongJ",
    author_email="songjie0613@126.com",
    description = "自适应密度峰值树聚类（Adaptive Density Peak Tree Clustering）",
    long_description = long_description,
    long_description_content_type="text/markdown",
    # url = "https://github.com/longweiqiang/dada_openapi_python",
    packages = setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)