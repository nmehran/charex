from setuptools import Extension, setup
import numpy as np


setup(
    ext_modules=[
        Extension(
            'charex._stringdtype',
            ['charex/_stringdtype.c'],
            include_dirs=[np.get_include()],
        ),
    ],
)
